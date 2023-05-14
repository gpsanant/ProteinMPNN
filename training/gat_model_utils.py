import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from model_utils import ProteinFeatures

def convert_adj_list_and_feature_to_pyg_edge_index_and_features(adj_list, adj_features):
    """ Convert adjacency list and features to PyG edge index and features """
    edge_index = [[], []]
    edge_features = []
    for i in range(len(adj_list)):
        for j in range(len(adj_list[i])):
            edge_index[0].append(i)
            edge_index[1].append(adj_list[i][j])
            edge_features.append(adj_features[i][j])
    edge_index = torch.tensor(edge_index, device=adj_features.device)
    edge_features = torch.stack(edge_features)
    return edge_index, edge_features
   

class ProtienGAT(nn.Module):
    def __init__(self, num_letters=21, num_node_features=128, num_edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super(ProtienGAT, self).__init__()

        # Hyperparameters
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.features = ProteinFeatures(num_node_features, num_edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(num_edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, index):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        # Just take the first sample? TODO: Address this
        E = E[index]
        E_idx = E_idx[index]
        S = S[index]
        mask = mask[index]
        chain_M = chain_M[index]
        residue_idx = residue_idx[index]
        chain_encoding_all = chain_encoding_all[index]
        # Convert adjacency list and features to PyG edge index and features
        edge_index, edge_features = convert_adj_list_and_feature_to_pyg_edge_index_and_features(E_idx, E)
        # node_features for the GATv2 is just a tensor for each node
        node_features = torch.zeros((E.shape[1], E.shape[-1]), device=E.device)
        # Create first GATv2
        encodingGAT = GATv2Conv(self.num_node_features, self.num_node_features, dropout=self.dropout, edge_dim=self.num_edge_features)
        # Run the GATv2
        node_features = encodingGAT((node_features, edge_features), edge_index)
        # Create sequence embeddings
        seq_features = self.W_s(S)
        # Concatenate sequence embeddings to node embeddings on last dimension
        node_seq_features = torch.cat((node_features, seq_features), dim=-1)
        # Concatenate 0ed out sequence embeddings to node embeddings on last dimension
        node_seq_features_masked = torch.cat((node_features, torch.zeros_like(seq_features)), dim=-1)
        # Generate sequence mask
        chain_M = chain_M*mask #update chain_M to include missing regions
        print("chain_M", chain_M.shape, "mask", mask.shape)
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        print("decoding_order", decoding_order.shape)
        mask_size = E_idx.shape[0]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        # Create masked node attr
        node_seq_features_masked = node_seq_features_masked * mask_fw + node_seq_features * mask_bw
        # Create second GATv2
        decodingGAT = GATv2Conv(self.node_features+self.hidden_dim, self.node_features, dropout=self.dropout, edge_dim=self.num_edge_features)
        # Run the GATv2
        output_node_attr = decodingGAT((node_seq_features_masked, edge_features), edge_index)
        # Run learned node embeddings through MLP to get logits
        logits = self.W_out(output_node_attr)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
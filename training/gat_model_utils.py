import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from model_utils import ProteinFeatures, cat_neighbors_nodes

def cat_neighbors_nodes_single_batch(node_features, edge_features_adj_list, E_idx):
    return cat_neighbors_nodes(torch.stack([node_features]), torch.stack([edge_features_adj_list]), torch.stack([E_idx])).squeeze(0)

def convert_adj_list_and_feature_to_pyg_edge_index_and_features(adj_list, adj_features):
    """ Convert adjacency list and features to PyG edge index and features """
    edge_index = [[], []]
    edge_features = []
    for i in range(len(adj_list)):
        for j in range(len(adj_list[i])):
            edge_index[0].append(i)
            edge_index[1].append(adj_list[i][j])
            edge_features.append(adj_features[i][j])
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=adj_features.device)
    edge_features = torch.stack(edge_features)
    return edge_index, edge_features
   

class ProtienGAT(nn.Module):
    def __init__(self, num_letters=21, num_node_features=128, num_edge_features=128,
        hidden_dim=128, vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1):
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
        # print("E", E.shape, "E_idx", E_idx.shape, "S", S.shape, "mask", mask.shape, "chain_M", chain_M.shape, "residue_idx", residue_idx.shape, "chain_encoding_all", chain_encoding_all.shape)
        # Just take the first sample? TODO: Address this
        E = E[index]
        E_idx = E_idx[index]
        S = S[index]
        mask = mask[index]
        chain_M = chain_M[index]
        residue_idx = residue_idx[index]
        chain_encoding_all = chain_encoding_all[index]
        edge_features_adj_list = self.W_e(E)
        # Convert adjacency list and features to PyG edge index and features
        edge_index, edge_features = convert_adj_list_and_feature_to_pyg_edge_index_and_features(E_idx, edge_features_adj_list)
        # node_features for the GATv2 is just a tensor for each node
        node_features = torch.zeros((E.shape[0], E.shape[-1]), device=E.device)
        # print("node_features", node_features.shape)
        # Create first GATv2
        encodingGAT = GATv2Conv(self.num_node_features, self.num_node_features, dropout=self.dropout, edge_dim=self.hidden_dim)
        # Run the GATv2
        node_features = encodingGAT(node_features, edge_index, edge_features)
        # Create sequence embeddings
        seq_features = self.W_s(S)
        # print("seq_features", seq_features.shape, "node_features", node_features.shape)
        edge_seq_features_adj_list = cat_neighbors_nodes_single_batch(seq_features, edge_features_adj_list, E_idx)
        edge_seq_features_adj_list_masked = cat_neighbors_nodes_single_batch(torch.zeros_like(seq_features), edge_features_adj_list, E_idx)
        # print("edge_seq_features_adj_list", edge_seq_features_adj_list.shape, "edge_seq_features_adj_list_masked", edge_seq_features_adj_list_masked.shape)
        # Generate sequence mask
        chain_M = chain_M*mask #update chain_M to include missing regions
        # print("chain_M", chain_M.shape, "mask", mask.shape)
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        # print("decoding_order", decoding_order.shape)
        mask_size = E_idx.shape[0]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        # print("permutation_matrix_reverse", permutation_matrix_reverse.shape)
        order_mask_backward = torch.einsum('ij, iq, jp->qp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        # print("order_mask_backward", order_mask_backward.shape)
        # print(order_mask_backward)
        mask_attend = torch.gather(order_mask_backward, 1, E_idx).unsqueeze(-1)
        # print("mask_attend", mask_attend.shape)
        # print(mask_attend)
        mask_1D = mask.view([mask.size(0), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        # Create masked edge seq features
        edge_seq_features_adj_list_masked = edge_seq_features_adj_list_masked * mask_fw + edge_seq_features_adj_list * mask_bw
        # convert to pyg edge index and features
        _, edge_seq_features = convert_adj_list_and_feature_to_pyg_edge_index_and_features(E_idx, edge_seq_features_adj_list)
        # Create second GATv2
        decodingGAT = GATv2Conv(self.num_node_features, self.num_node_features, dropout=self.dropout, edge_dim=self.num_edge_features+self.hidden_dim)
        # Run the GATv2
        output_node_features = decodingGAT(node_features, edge_index, edge_seq_features)
        # Run learned node embeddings through MLP to get logits
        logits = self.W_out(output_node_features)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
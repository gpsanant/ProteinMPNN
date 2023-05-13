import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2
from model_utils import ProteinFeatures

class ProtienGAT(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super(ProtienGAT, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        # Create first GATv2
        encodingGAT = GATv2(self.node_features, self.node_features, dropout=0.1, edge_dim=self.edge_features)
        # Run h_V and h_E through the GATv2
        h_V = encodingGAT(h_V, 
        # Add sequnce embeddings to h_E to get h_ES
        # Add 0ed out sequence embeddings to h_E to get masked sequence and edge embeddings
        # Concatenate sequence and edge embeddings to vertices to get new edge edges features
        # Run h_V and new edge features through GATv2
        # Run learned node embeddings through MLP to get logits
        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
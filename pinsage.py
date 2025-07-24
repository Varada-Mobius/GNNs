import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data # Import Data from torch_geometric.data
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score
import random

class ImportanceBasedSampler:
    """Importance-based neighbor sampler for PinSAGE"""

    def __init__(self, edge_index, num_nodes, walk_length=5, num_walks=50):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.walk_length = walk_length
        self.num_walks = num_walks

        # Convert to NetworkX for easier random walk computation
        self.G = to_networkx(
            Data(edge_index=edge_index, num_nodes=num_nodes), # Use Data from torch_geometric.data
            to_undirected=True
        )

        # Precompute importance scores using random walks
        self.importance_scores = self._compute_importance_scores()

    def _random_walk(self, start_node, length):
        """Perform a random walk starting from start_node"""
        if start_node not in self.G:
            return [start_node]

        walk = [start_node]
        current = start_node

        for _ in range(length - 1):
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            walk.append(current)

        return walk

    def _compute_importance_scores(self):
        """Compute importance scores for all node pairs using random walks"""
        importance = {}

        for node in range(self.num_nodes):
            node_importance = {}
            visit_counts = {}

            # Perform multiple random walks from this node
            for _ in range(self.num_walks):
                walk = self._random_walk(node, self.walk_length)
                for visited_node in walk[1:]:  # Exclude the starting node
                    visit_counts[visited_node] = visit_counts.get(visited_node, 0) + 1

            # Normalize visit counts to get importance scores
            total_visits = sum(visit_counts.values())
            if total_visits > 0:
                for neighbor, count in visit_counts.items():
                    node_importance[neighbor] = count / total_visits

            importance[node] = node_importance

        return importance

    def sample_neighbors(self, nodes, num_samples=10):
        """Sample important neighbors for given nodes"""
        sampled_neighbors = {}
        importance_weights = {}

        for node in nodes:
            node = int(node)
            if node in self.importance_scores:
                # Get neighbors and their importance scores
                neighbors_scores = self.importance_scores[node]

                if neighbors_scores:
                    # Sort neighbors by importance and take top k
                    sorted_neighbors = sorted(
                        neighbors_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # Sample up to num_samples neighbors
                    selected = sorted_neighbors[:min(num_samples, len(sorted_neighbors))]

                    if selected:
                        neighbors, weights = zip(*selected)
                        sampled_neighbors[node] = list(neighbors)
                        importance_weights[node] = list(weights)
                    else:
                        sampled_neighbors[node] = []
                        importance_weights[node] = []
                else:
                    sampled_neighbors[node] = []
                    importance_weights[node] = []
            else:
                sampled_neighbors[node] = []
                importance_weights[node] = []

        return sampled_neighbors, importance_weights
    
class ImportancePooling(nn.Module):
    """Learnable importance-weighted pooling for PinSAGE"""

    def __init__(self, in_channels):
        super().__init__()
        self.importance_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1)
        )

    def forward(self, x, neighbor_indices, importance_weights):
        """
        x: node features [num_nodes, in_channels]
        neighbor_indices: dict mapping node_idx -> list of neighbor indices
        importance_weights: dict mapping node_idx -> list of importance weights
        """
        batch_size = len(neighbor_indices)
        out = torch.zeros(batch_size, x.size(1), device=x.device)

        for i, (node_idx, neighbors) in enumerate(neighbor_indices.items()):
            if len(neighbors) == 0:
                # If no neighbors, use self features
                out[i] = x[node_idx] if isinstance(node_idx, int) else x[0]
                continue

            # Get neighbor features
            neighbor_features = x[neighbors]  # [num_neighbors, in_channels]

            # Compute learnable importance scores
            learned_importance = self.importance_mlp(neighbor_features)  # [num_neighbors, 1]
            learned_importance = torch.softmax(learned_importance.squeeze(-1), dim=0)

            # Combine with pre-computed importance weights
            if node_idx in importance_weights and importance_weights[node_idx]:
                precomputed_weights = torch.tensor(
                    importance_weights[node_idx],
                    device=x.device,
                    dtype=torch.float
                )
                # Normalize precomputed weights
                precomputed_weights = torch.softmax(precomputed_weights, dim=0)

                # Combine learned and precomputed importance
                final_weights = 0.5 * learned_importance + 0.5 * precomputed_weights
            else:
                final_weights = learned_importance

            # Weighted aggregation
            out[i] = torch.sum(neighbor_features * final_weights.unsqueeze(-1), dim=0)

        return out
class PinSAGELayer(nn.Module):
    """Single PinSAGE layer with importance-based sampling and pooling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Self and neighbor transformation
        self.self_transform = nn.Linear(in_channels, out_channels)
        self.neighbor_transform = nn.Linear(in_channels, out_channels)

        # Importance-based pooling
        self.importance_pooling = ImportancePooling(in_channels) # Pooling still operates on original feature dimension

        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x, sampled_neighbors, importance_weights, batch_nodes):
        """
        x: all node features
        sampled_neighbors: dict of sampled neighbors for batch nodes
        importance_weights: importance weights for sampled neighbors
        batch_nodes: nodes in current batch
        """
        # Transform self features
        self_features = self.self_transform(x[batch_nodes])

        # Aggregate neighbor features using importance pooling
        # Importance pooling outputs dimension of in_channels, then transformed
        neighbor_features = self.importance_pooling(x, sampled_neighbors, importance_weights)
        neighbor_features = self.neighbor_transform(neighbor_features)


        # Combine self and neighbor features
        out = self_features + neighbor_features
        out = self.layer_norm(out)

        return out


class PinSAGE(nn.Module):
    """PinSAGE model for node classification"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_samples=10):
        super().__init__()
        self.num_layers = num_layers
        self.num_samples = num_samples

        # Build layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(PinSAGELayer(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(PinSAGELayer(hidden_channels, hidden_channels))

        # Output layer
        if num_layers > 1:
            self.layers.append(PinSAGELayer(hidden_channels, out_channels))
        else:
            # If only one layer, it's the output layer
            self.layers[0] = PinSAGELayer(in_channels, out_channels)


    def forward(self, x, sampler, batch_nodes):
        """
        x: all node features
        sampler: ImportanceBasedSampler instance
        batch_nodes: nodes to compute embeddings for
        """
        current_features = x  # Start with original features
        current_nodes = batch_nodes

        for i, layer in enumerate(self.layers):
            # Sample neighbors for current batch
            sampled_neighbors, importance_weights = sampler.sample_neighbors(
                current_nodes, self.num_samples
            )

            # Apply PinSAGE layer
            x_batch = layer(current_features, sampled_neighbors, importance_weights, current_nodes)

            # Update features for batch nodes
            if i < len(self.layers) - 1:
                # Apply activation and update features for intermediate layers
                new_features = torch.zeros(current_features.size(0), x_batch.size(1), device=x.device)
                new_features[current_nodes] = F.relu(x_batch)
                current_features = new_features
            else:
                # Final layer - no activation, return the output
                final_output = x_batch

        return final_output

def train_pinsage_cora():
    # Load Cora dataset
    dataset = Planetoid(root='data/', name='Cora', transform=ToUndirected())
    data = dataset[0]

    print(f"Data: {data}")
    print(f"Number of nodes: {data.num_nodes}, edges: {data.num_edges}")
    print(f"Number of features: {data.num_node_features}")
    print(f"Number of classes: {dataset.num_classes}")

    # Initialize importance-based sampler
    print("Computing importance scores...")
    sampler = ImportanceBasedSampler(
        data.edge_index,
        data.num_nodes,
        walk_length=3,
        num_walks=20  # Reduced for faster computation on Cora
    )

    # Initialize model
    model = PinSAGE(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
        num_samples=5  # Sample top 5 important neighbors
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()

        # Forward pass on training nodes
        train_nodes = data.train_mask.nonzero().squeeze().tolist()
        if isinstance(train_nodes, int):
            train_nodes = [train_nodes]

        out = model(data.x, sampler, train_nodes)
        loss = criterion(out, data.y[train_nodes])

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Test on all nodes (in practice, you'd batch this)
        test_nodes = data.test_mask.nonzero().squeeze().tolist()
        if isinstance(test_nodes, int):
            test_nodes = [test_nodes]

        test_out = model(data.x, sampler, test_nodes)
        pred = test_out.argmax(dim=1)

        test_acc = accuracy_score(
            data.y[test_nodes].cpu().numpy(),
            pred.cpu().numpy()
        )

        print(f'Test Accuracy: {test_acc:.4f}')

    return model, sampler


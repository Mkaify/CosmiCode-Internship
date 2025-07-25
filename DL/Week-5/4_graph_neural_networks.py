"""
# Graph Neural Network Tutorial

This notebook demonstrates how to implement Graph Neural Networks (GNNs) for **node classification** using **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)**.
We will use the **Cora** dataset from PyTorch Geometric.

**Key goals**:
1. Implement model definitions (GCN, GAT) in a modular fashion.
2. Create reusable training and evaluation functions.
3. Load and preprocess graph data using up-to-date libraries.
4. Run end-to-end training and report accuracy.

"""
import sys

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv


"""
2. Data Loading and Preprocessing
We will load the *Cora* dataset and prepare a DataLoader.
"""

dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# For full-batch training, we use the single graph directly
print(f"Dataset: {dataset.name}")
print(data)

"""
3. Modular Model Definitions
Define two models: *GCNNet* and *GATNet*.
"""

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # concatenate heads -> hidden_channels * heads
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

"""
4. Training and Evaluation Functions
"""

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].argmax(dim=1)
        acc = int((pred == data.y[mask]).sum()) / int(mask.sum())
        accs.append(acc)
    return accs

"""
5. Experiments
Train and evaluate both GCN and GAT on Cora.
"""

devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', devices)

results = {}
for name, Model in [('GCN', GCNNet), ('GAT', GATNet)]:
    torch.manual_seed(42)
    model = Model(dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes).to(devices)
    data = data.to(devices)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_test = 0, 0
    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if epoch % 50 == 0:
            print(f"{name} Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
    results[name] = (best_val, best_test)

print("\n## Final Results ##")
for name, (val_acc, test_acc) in results.items():
    print(f"{name}: Best Val Acc: {val_acc:.4f}, Test Acc @ Best Val: {test_acc:.4f}")
    
    

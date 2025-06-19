# model_definition.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
import timm

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CNNModel, self).__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(self.feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = self.bn(features)
        features = self.dropout(features)
        return features

# GNN Model
class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_channels, num_classes, dropout_rate=0.3):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + input_size, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_orig = x
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = self.bn3(self.conv3(x, edge_index))
        x_pooled = global_mean_pool(x, batch)
        x_orig_pooled = global_mean_pool(x_orig, batch)
        x_combined = torch.cat([x_pooled, x_orig_pooled], dim=1)
        return self.classifier(x_combined)

# Combined CNN-GNN
class CNN_GNN_Model(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(CNN_GNN_Model, self).__init__()
        self.cnn = CNNModel(dropout_rate)
        self.feature_dim = self.cnn.feature_dim
        self.gnn = GNNModel(input_size=self.feature_dim, hidden_channels=256,
                            num_classes=num_classes, dropout_rate=dropout_rate)

    def construct_adaptive_knn_graph(self, features, k=7):
        batch_size = features.size(0)
        if batch_size == 1:
            return torch.tensor([[0], [0]], device=features.device)
        adaptive_k = min(max(3, k), batch_size - 1)
        dist = torch.cdist(features, features)
        noise = torch.randn_like(dist) * 1e-6
        dist = dist + noise
        knn_idx = dist.topk(k=adaptive_k + 1, largest=False).indices[:, 1:]
        edge_list = []
        for i in range(batch_size):
            for j in knn_idx[i]:
                if i != j.item():
                    edge_list.append([i, j.item()])
        if not edge_list:
            edge_list = [[i, (i + 1) % batch_size] for i in range(batch_size)]
        edge_index = torch.tensor(edge_list).t().to(features.device)
        return edge_index

    def forward(self, images):
        features = self.cnn(images)
        edge_index = self.construct_adaptive_knn_graph(features)
        batch = torch.arange(images.size(0), dtype=torch.long, device=images.device)
        data = Data(x=features, edge_index=edge_index, batch=batch)
        return self.gnn(data)

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import networkx as nx
import math
import numpy as np

class FiLMGnn(nn.Module):
  def __init__(self, node_feat_dim, gnn_dim, edge_feat_dim, num_layers=6):
    super().__init__()
    self.gnn_dim = gnn_dim
    self.num_layers = num_layers
    self.node_embed = nn.Linear(node_feat_dim, gnn_dim)
    self.convs = nn.ModuleList()
    self.layer_norms = nn.ModuleList()
    for _ in range(num_layers):
      self.layer_norms.append(nn.LayerNorm(gnn_dim))
      # 增加 heads 到 4，提升模型表达能力
      self.convs.append(
        pyg_nn.GATConv(gnn_dim, gnn_dim, heads=4, concat=False, edge_dim=edge_feat_dim)
      )
    self.edge_output_head = nn.Sequential(
      nn.Linear(gnn_dim * 2 + edge_feat_dim, gnn_dim),
      nn.ReLU(),
      nn.Linear(gnn_dim, 1)
    )
    
    self.register_buffer('gamma_neutral', torch.ones(num_layers, gnn_dim))
    self.register_buffer('beta_neutral', torch.zeros(num_layers, gnn_dim))

  def forward(self, data, gamma=None, beta=None, return_node_feats=False):

    if gamma is None: 
      gamma = self.gamma_neutral
    if beta is None: 
      beta = self.beta_neutral

    h = self.node_embed(data.x)

    for l in range(self.num_layers):
      h_norm = self.layer_norms[l](h)
      
      gamma_l = gamma[l]
      beta_l = beta[l]
      if gamma_l.dim() == 1: 
        gamma_l = gamma_l.unsqueeze(0)
        beta_l = beta_l.unsqueeze(0)

      h_modulated = gamma_l * h_norm + beta_l
      h = self.convs[l](h_modulated, data.edge_index, edge_attr=data.edge_attr)
      h = h.relu()
      
    h_src = h[data.edge_index[0]]
    h_dst = h[data.edge_index[1]]
    edge_features = torch.cat([h_src, h_dst, data.edge_attr], dim=-1)
    
    edge_logits = self.edge_output_head(edge_features).squeeze()
    
    if return_node_feats:
      return edge_logits, h  # 返回 logits 和 节点特征
    else:
      return edge_logits     # 仅返回 logits (兼容旧代码)


      
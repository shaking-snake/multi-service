# 文件路径: MS/A2C/ActorCritic.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn.glob import global_mean_pool

from ..LSTM.PreferenceModule import PreferenceModule
from ..GNN.Pretrain.DijkstraGnn import GNNPretrainModel

class ActorCritic(nn.Module):
  """
  E2E FiLM-GNN 模型 (进阶 Critic 版)
  """
  def __init__(self, config):        
    super().__init__()
    
    self.gnn_layers = config.gnn_layers
    self.gnn_hidden_dim = config.gnn_hidden_dim

    # 主体初始化 (保持不变)
    self.lstm_body = PreferenceModule(config.lstm_input_dim, config.lstm_hidden_dim, config.lstm_layers)
    self.gnn_model = GNNPretrainModel(config.gnn_node_dim, config.gnn_hidden_dim, config.gnn_edge_dim, config.gnn_layers)

    # 加载权重
    if config.pretrained_lstm_path:
      print(f"[ms] Loading LSTM: {config.pretrained_lstm_path}")
      self.lstm_body.load_state_dict(torch.load(config.pretrained_lstm_path, map_location='cpu'))
        
    if config.pretrained_gnn_path:
      print(f"[ms] Loading GNN: {pretrained_gnn_path}")
      gnn_state = torch.load(config.pretrained_gnn_path, map_location='cpu')
      gnn_state = {k.replace('module.', ''): v for k, v in gnn_state.items()}
      self.gnn_model.load_state_dict(gnn_state, strict=False)

    # 1. FiLM Generator
    self.total_film_params = gnn_layers * gnn_hidden_dim * 2
    self.film_generator = nn.Sequential(
      nn.Linear(lstm_hidden_dim, gnn_hidden_dim),
      nn.ReLU(),
      nn.Linear(gnn_hidden_dim, self.total_film_params)
    )

    # 初始化技巧：让初始输出接近0，即gamma=1, beta=0
    with torch.no_grad():
      self.film_generator[-1].weight.data *= 0.01
      self.film_generator[-1].bias.data.fill_(0.0)

    # 2. Actor Head (重置后的 GNN 输出头)
    self.gnn_model.edge_output_head = nn.Sequential(
      nn.Linear(gnn_hidden_dim * 2 + gnn_edge_dim, gnn_hidden_dim),
      nn.ReLU(),
      nn.Linear(gnn_hidden_dim, 1)
    )

    # 3. [进阶] Critic Head
    # 输入维度 = LSTM流特征(128) + GNN图特征(256)
    # 这让 Critic 拥有了“上帝视角”：既知道业务需求，又知道网络拥塞状况
    self.critic_head = nn.Sequential(
      nn.Linear(lstm_hidden_dim + gnn_hidden_dim, gnn_hidden_dim),
      nn.ReLU(),
      nn.Linear(gnn_hidden_dim, 1)
    )

    # --- 冻结主体 ---
    for p in self.lstm_body.parameters(): p.requires_grad = False
    for p in self.gnn_model.node_embed.parameters(): p.requires_grad = False
    for p in self.gnn_model.convs.parameters(): p.requires_grad = False
    for p in self.gnn_model.layer_norms.parameters(): p.requires_grad = False

  def forward(self, flow_fingerprint, graph_data):
    # 1. LSTM 提取流特征
    h_flow = self.lstm_body(flow_fingerprint) # (B, 128)

    # 2. FiLM 生成参数
    film_flat = self.film_generator(h_flow)
    # reshape: (B, L, 2, D)
    film_params = film_flat.view(-1, self.gnn_layers, 2, self.gnn_hidden_dim)
    
    # 为了适配 PyG，我们假设 B=1 (RL 的标准做法)
    # gamma/beta: (L, D)
    gamma = 1.0 + film_params[0, :, 0, :] 
    beta  = film_params[0, :, 1, :]

    # 3. GNN 提取图特征 & Actor 输出
    # [关键] 调用修改后的 GNN，获取 node_features (h_graph_raw)
    edge_logits, node_features = self.gnn_model(
      graph_data, 
      gamma=gamma, 
      beta=beta, 
      return_node_feats=True # 必须为 True
    )
    
    # Actor 策略分布
    dist = Categorical(logits=edge_logits)

    # 4. [进阶] Critic 计算价值
    # Global Mean Pooling: 将所有节点的特征平均，得到“整张图”的特征向量
    # node_features: (Num_Nodes, 256) -> h_graph: (B, 256)
    h_graph = global_mean_pool(node_features, graph_data.batch)
    
    # 拼接：流需求 + 图状态
    # 形状: (B, 128+256) = (B, 384)
    state_fusion = torch.cat([h_flow, h_graph], dim=1)
    
    # 预测价值
    value = self.critic_head(state_fusion)

    return dist, value.squeeze(1)
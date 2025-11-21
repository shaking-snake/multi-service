import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn.glob import global_mean_pool

# 导入我们已经训练好的两个模型
from ..LSTM.PreferenceModule import PreferenceModule
from ..GNN.Pretrain.DijkstraGnn import GNNPretrainModel

class ActorCritic(nn.Module):
  """
  E2E FiLM-GNN 模型，同时作为 A2C 框架的 Actor 和 Critic。
  """
  def __init__(self, 
    # LSTM (偏好模块) 参数
    lstm_input_dim=2, 
    lstm_hidden_dim=128, 
    lstm_layers=2,
    
    # GNN (路径选择模块) 参数
    gnn_hidden_dim=256, # gnn 隐藏特征数
    gnn_node_dim=10,    # 节点特征数
    gnn_edge_dim=5,     # 边 特征数
    gnn_layers=6,       # gnn 层数
    
    # 预训练权重路径
    pretrained_lstm_path="MS/LSTM/pretrained-model.pth",         # 预训练的 lstm 模型
    pretrained_gnn_path="MS/GNN/gnn_pretrained_model.pth"        # 预训练的 gnn 模型(Dijkstra)
    ):        
      
    super().__init__()
    
    self.gnn_hidden_dim = gnn_hidden_dim
    self.gnn_layers = gnn_layers

    # ======================================================================
    # 1. 实例化两个“主体”
    # ======================================================================
    
    # 偏好模块 (LSTM Body)
    self.lstm_body = LstmLayer(lstm_input_dim, lstm_hidden_dim, lstm_layers)
    
    # 路径选择模块 (GNN Model) - 我们先加载整个模型，包括预训练的头
    self.gnn_model = FiLMGnnModel(
      gnn_node_dim, gnn_hidden_dim, gnn_edge_dim, gnn_layers
    )

    # ======================================================================
    # 2. 加载预训练权重 (关键步骤)
    # ======================================================================
    
    if pretrained_lstm_path:
      print(f"🔄 正在加载 [LSTM Body] 权重来源: {pretrained_lstm_path}")
      # 加载 jstm 模型
      lstm_state = torch.load(pretrained_lstm_path, map_location='cpu')
      self.lstm_body.load_state_dict(lstm_state)
        
    if pretrained_gnn_path:
      print(f"🔄 正在加载 [GNN Body] 权重来源: {pretrained_gnn_path}")
      gnn_state = torch.load(pretrained_gnn_path, map_location='cpu')
      
      # [关键] 移除多卡训练时 DataParallel 自动添加的 'module.' 前缀
      gnn_state_cleaned = {k.replace('module.', ''): v for k, v in gnn_state.items()}
      
      # 加载权重，strict=False 允许我们稍后覆盖 GNN 头
      self.gnn_model.load_state_dict(gnn_state_cleaned, strict=False)

    # ======================================================================
    # 3. 定义新的“头” (随机初始化)
    # ======================================================================
    
    # [新头 1] FiLM 生成器 (缝合模块 / RNN 头)
    # 目标: (B, D_lstm) -> (B, L*D_gnn*2)
    self.total_film_params = gnn_layers * gnn_hidden_dim * 2
    self.film_generator = nn.Sequential(
      nn.Linear(lstm_hidden_dim, gnn_hidden_dim),
      nn.ReLU(),
      nn.Linear(gnn_hidden_dim, self.total_film_params)
      # 最后一层没有激活函数，允许 gamma/beta 取任意值
    )

    # [新头 2] 路径输出头 (GNN 头 / Actor Head)
    # 按照项目要求，重新随机初始化 GNN 头
    # 它将取代 gnn_model 中预训练好的那个头
    self.gnn_model.edge_output_head = nn.Sequential(
      nn.Linear(gnn_hidden_dim * 2 + gnn_edge_dim, gnn_hidden_dim),
      nn.ReLU(),
      nn.Linear(gnn_hidden_dim, 1)
    )
    print("✅ GNN 预训练头已替换为随机初始化的 Actor 头。")

    # [新头 3] 价值评估头 (Critic Head)
    # 评估 V(s)，输入是流摘要和图摘要的拼接
    self.critic_head = nn.Sequential(
      nn.Linear(lstm_hidden_dim + gnn_hidden_dim, gnn_hidden_dim),
      nn.ReLU(),
      nn.Linear(gnn_hidden_dim, 1) # 输出一个标量价值
    )

    # ======================================================================
    # 4. 冻结主体 (关键步骤)
    # ======================================================================
    # 按照项目要求，我们只训练新初始化的“头”
    
    for param in self.lstm_body.parameters():
      param.requires_grad = False
        
    # 冻结 GNN 的嵌入层、卷积层、归一化层
    for param in self.gnn_model.node_embed.parameters():
      param.requires_grad = False
    for param in self.gnn_model.convs.parameters():
      param.requires_grad = False
    for param in self.gnn_model.layer_norms.parameters():
      param.requires_grad = False
        
    # 注意：self.gnn_model.edge_output_head (Actor头) 仍然是可训练的
    
    print("🔒 [主体] LSTM Body 和 GNN Body 已冻结。")
    print("🔓 [新头] FiLM 生成器、Actor 头、Critic 头 保持可训练。")


  def forward(self, flow_fingerprint, graph_data):
    """
    E2E 前向传播。在 RL 循环中，通常 B=1。
    :param flow_fingerprint: (B, N, C) - e.g., (1, 50, 2)
    :param graph_data: PyG Batch 对象 (包含一张图)
    :return: (dist, value) - 动作分布, 状态价值
    """
      
    # --- 1. 偏好模块 (LSTM) ---
    # (B, N, C) -> (B, D_lstm)
    h_n = self.lstm_body(flow_fingerprint) 
      
    # --- 2. 缝合模块 (FiLM 生成器) ---
    # (B, D_lstm) -> (B, L*D_gnn*2)
    film_params_flat = self.film_generator(h_n)
      
    # 重塑参数
    batch_size = h_n.size(0)
    film_params = film_params_flat.view(
      batch_size, self.gnn_layers, 2, self.gnn_hidden_dim
    )
    # (B, L, D_gnn)
    gamma = film_params[:, :, 0, :] 
    beta  = film_params[:, :, 1, :]
    
    # [关键] RL 通常 B=1，我们去掉 Batch 维度以匹配 GNN 的 forward 签名
    if batch_size != 1:
      raise NotImplementedError("RL 循环的 ActorCritic 目前只支持 B=1")
      
    gamma_squeezed = gamma.squeeze(0) # (L, D_gnn)
    beta_squeezed = beta.squeeze(0)  # (L, D_gnn)
      
    # --- 3. 路径选择模块 (Actor) ---
    # GNN 接收动态生成的 FiLM 参数
    #
    edge_logits, node_features = self.gnn_model(
      graph_data, 
      gamma=gamma_squeezed, 
      beta=beta_squeezed, 
      return_node_feats=True                    # 请求 GNN 返回节点特征
    )
    
    # [Actor 输出]: 将 logits 转换为概率分布
    # edge_logits 形状 (Num_Edges,)
    # 这是 Actor 的策略 (Policy)
    dist = Categorical(logits=edge_logits)

    # --- 4. 价值评估模块 (Critic) ---
    # 我们需要一个图级别的表示。使用全局平均池化。
    # node_features 形状 (Num_Nodes, D_gnn)
    # graph_data.batch 告诉池化函数哪些节点属于哪个图（即使只有一个图）
    graph_embedding = global_mean_pool(node_features, graph_data.batch) # (B, D_gnn)
    
    # 拼接流摘要和图摘要
    state_embedding = torch.cat([h_n, graph_embedding], dim=1)          # (B, D_lstm + D_gnn)
    
    # [Critic 输出]: 评估当前状态的价值 V(s)
    value = self.critic_head(state_embedding) # (B, 1)
    
    return dist, value.squeeze(1) # 返回分布和标量价值
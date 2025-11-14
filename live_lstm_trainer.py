import os
import sys
import networkx as nx
import numpy as np
import time
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from functools import partial
# --- 导入你的环境文件 ---
from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.FlowGenerator import FlowGenerator, FlowType
from MS.Env.MininetController import get_flow_fingerprint, measure_path_qos

# --- 导入 PyTorch 和你的 RNN 分类器 ---
import torch
import torch.nn as nn
import torch.optim as optim
from MS.LSTM.Pretrain.TmpClassifier import RNNClassifier #
from tqdm import tqdm # 用于显示进度条

# -----------------------------------------------
# GraphTopo 类 (与 tmp_runner.py 中修复后的一致)
# -----------------------------------------------
class GraphTopo(Topo):
  def __init__(self, blueprint_g: nx.Graph, **opts):
    Topo.__init__(self, **opts)
    
    for node_id in blueprint_g.nodes():
      self.addSwitch(f's{node_id}', protocols='OpenFlow13')
      self.addHost(f'h{node_id}')
      self.addLink(f'h{node_id}', f's{node_id}', bw=1000, delay='0.1ms')

    for u, v, data in blueprint_g.edges(data=True):
      bw = data.get('bandwidth', 1000)
      delay = f"{data.get('delay', 1)}ms"
      self.addLink(f's{u}', f's{v}', bw=bw, delay=delay)

# -----------------------------------------------
# 核心训练逻辑
# -----------------------------------------------
def run_live_training():
  """主运行函数"""
  net = None
  
  # --- 训练参数 ---
  TRAINING_STEPS = 500  # 我们总共生成 500 个样本
  LEARNING_RATE = 0.001
  
  # --- 模型参数 (!! 关键 !!) ---
  # 根据你上次的 "size mismatch" 错误日志，
  # 我们知道你的 Classify-model.pth 是用这些参数训练的：
  INPUT_DIM = 2           # 2个参数 (delay, IAT)
  RNN_LAYERS = 2          # RNN 层数
  OUTPUT_DIM = 3          # 3个类别 (VOIP, STREAMING, INTERACTIVE)
  HIDDEN_DIM = 128        # 
  
  # 类别映射 (FlowType.value 是 1-based, 损失函数是 0-based)
  # FlowType.VOIP.value = 1 -> 索引 0
  # FlowType.STREAMING.value = 2 -> 索引 1
  # FlowType.INTERACTIVE.value = 3 -> 索引 2
  
  # --- 初始化 PyTorch 组件 ---
  info(f"*** 正在初始化模型 (HIDDEN_DIM={HIDDEN_DIM}, OUTPUT_DIM={OUTPUT_DIM})\n")
  model = RNNClassifier(
    input_dim=INPUT_DIM, 
    rnn_hidden_dim=HIDDEN_DIM, 
    num_classes=OUTPUT_DIM, 
    rnn_layers=RNN_LAYERS
    )
  
  # (重要: 确保 TmpClassifier.py 里的 intermediate_dim 已被修复为 64)
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  
  # 将模型置于训练模式
  model.train()

  try:
    # --- 步骤 1: 生成拓扑 ---
    print("--- 步骤 1: 生成拓扑 [NetworkGenerator] ---")
    topo_gen = TopologyGenerator(num_nodes_range=(5, 8))
    g = topo_gen.generate_topology()
    print(f"拓扑生成完毕: {len(g.nodes())} 个节点, {len(g.edges())} 条边。")

    # --- 步骤 2: 启动 Mininet ---
    print("\n--- 步骤 2: 启动 Mininet ---")
    RemoteCtrl = partial(RemoteController, ip='127.0.0.1', port=6633)

    net = Mininet(
      topo=GraphTopo(g),
      switch=OVSKernelSwitch,
      link=TCLink,
      controller=RemoteCtrl
    )

    net.start()
    
    # --- 步骤 2.B: 测试 Mininet 连通性 (pingAll) ---
    print("\n--- 步骤 2.B: 测试 Mininet 连通性 (pingAll) ---")
    packet_loss_percentage = net.pingAll()

    # --- 步骤 3: 实例化生成器 ---
    flow_gen = FlowGenerator()

    info(f"*** 步骤 4: 开始在线训练 (共 {TRAINING_STEPS} 步)\n")
    
    # ----------------------------------------------------
    # 这就是 "在线训练" 循环
    # 它取代了 "for data in dataloader:"
    # ----------------------------------------------------
    pbar = tqdm(range(TRAINING_STEPS))
    total_loss = 0.0
    correct_predictions = 0

    for i in pbar:
      # A. 生成一个新样本 (数据+标签)
      # --------------------------
      flow_type, flow_profile = flow_gen.get_random_flow()
      # 【关键】标签：FlowType.value (1,2,3) -> 损失函数 (0,1,2)
      label = flow_type.value - 1 
      label_tensor = torch.tensor([label], dtype=torch.long) #

      # B. 获取 Mininet 中的主机
      # --------------------------
      s_id, d_id = topo_gen.select_source_destination()
      S_host = net.get(f'h{s_id}')
      D_host = net.get(f'h{d_id}')
      
      # C. 在 Mininet 中"实时"生成指纹 (数据)
      # ---------------------------------
      fingerprint_matrix = get_flow_fingerprint(S_host, D_host, flow_profile) #
      
      if fingerprint_matrix is None:
        info(f"警告: 第 {i} 步指纹生成失败，跳过。\n")
        continue
    
      # D. 准备 PyTorch 张量
      # --------------------
      # (50, 2) -> (1, 50, 2) (添加 Batch 维度)
      input_tensor = torch.from_numpy(fingerprint_matrix).unsqueeze(0) 
      
      # E. 执行训练步骤
      # -----------------
      optimizer.zero_grad()            # 清空梯度
      logits = model(input_tensor)     # 前向传播
      loss = criterion(logits, label_tensor) # 计算损失
      loss.backward()                  # 反向传播
      optimizer.step()                 # 更新权重
      
      # F. 记录统计数据
      # -----------------
      total_loss += loss.item()
      predicted_index = torch.argmax(logits, dim=1).item()
      if predicted_index == label:
          correct_predictions += 1
      
      if (i+1) % 50 == 0:
          avg_loss = total_loss / 50
          accuracy = correct_predictions / 50
          info(f"\n[步骤 {i+1}/{TRAINING_STEPS}] 平均损失: {avg_loss:.4f}, 准确率: {accuracy*100:.2f}%\n")
          total_loss = 0.0
          correct_predictions = 0

    info(f"*** 训练完成 ***\n")
    
    # --- 步骤 5: 保存新训练的模型 ---
    output_model_path = "./lstm_live_trained.pth"
    torch.save(model.state_dict(), output_model_path)
    info(f"新模型已保存到: {output_model_path}\n")

  
  except Exception as e:
      info(f"\n--- 仿真出错 ---")
      info(f"{e}\n")
      import traceback
      traceback.print_exc()
      
  finally:
      # 确保 Mininet 总是被停止
      if net:
          info("\n*** 步骤 6: 停止 Mininet ***\n")
          net.stop()
      
      # 确保 Mininet 被彻底清理
      info("INFO: 运行 mn -c 以防万一...\n")
      os.system('sudo mn -c')


if __name__ == '__main__':
  setLogLevel('info')
  
  if os.getuid() != 0:
    print("错误：此脚本必须以 root 权限 (sudo) 运行。")
  else:
    run_live_training()
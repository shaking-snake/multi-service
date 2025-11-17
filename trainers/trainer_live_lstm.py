import os
import sys
import networkx as nx
import numpy as np
import time
from functools import partial
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.log import setLogLevel, info
import torch.optim.lr_scheduler as lr_scheduler
from mininet.node import OVSKernelSwitch, RemoteController
# --- 导入你的环境文件 ---
from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.FlowGenerator import FlowGenerator, FlowType
from MS.Env.MininetController import get_a_mininet, send_packet_and_capture

# --- 导入 PyTorch 和你的 RNN 分类器 ---
import torch
import torch.nn as nn
import torch.optim as optim
from MS.LSTM.Pretrain.TmpClassifier import RNNClassifier #
from tqdm import tqdm # 用于显示进度条


def run_live_training():
  
  # 训练参数
  LEARNING_RATE = 4e-3
  TOTAL_BATCH = 200
  ACCUMULATION_STEPS = 128  # 梯度累积
  TRAINING_STEPS = ACCUMULATION_STEPS*TOTAL_BATCH  # 我们总共生成 128*200 个样本
  
  Lstm_PATH       = "./trained_model/trained_lstm.pth"
  Classifier_PATH = "./trained_model/trained_classifier.pth"
  
  # 模型参数 
  INPUT_DIM = 2            # 2个参数 (delay, IAT)
  RNN_LAYERS = 2           # RNN 层数
  NUM_CLASSES = 3          # 3个类别 (VOIP, STREAMING, INTERACTIVE)
  HIDDEN_DIM  = 256        # 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # 初始化 PyTorch 组件

  model = RNNClassifier(
    input_dim=INPUT_DIM, 
    rnn_hidden_dim=HIDDEN_DIM, 
    num_classes=NUM_CLASSES, 
    rnn_layers=RNN_LAYERS
    )
  
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  """
    mode='min' : 我们希望监控的指标(loss)越小越好
    factor=0.5 : 触发调整时，新LR = 旧LR * 0.5
    patience=3: 如果连续 10 次更新（即 10 * 128 个流），Loss 都没有明显下降，才触发减少
    verbose=True: 触发时打印日志
  """

  scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=10
  )

  # 将模型置于训练模式
  model.train()

  # 流生成参数
  FLOW_DURATION = 30          # 最长发包时间
  N_PACKETS_TO_CAPTURE = 30   # 抓包数量

  """
  为了尽可能快速的获取包，采用简单的拓扑：只包含两个节点 h1，h2
  """

  g = nx.Graph()
  g.add_nodes_from(["1", "2"])
  g.add_edge("1", "2")

  try:
    with get_a_mininet(g) as net:
      print("======网络已启动=======")
      print(net.hosts)
      net.pingAll()
      
      server, client = net.get("h1", "h2")

      flow_gen = FlowGenerator()   #实例化生成器 

      info(f"==== 开始在线训练 (共 {TRAINING_STEPS} 个数据), 每批次 {ACCUMULATION_STEPS} 个数据=====")
      info(f"==== 使用设备：{device}")
      pbar = tqdm(range(TRAINING_STEPS))
      correct_predictions = 0
      total_loss = 0.0
      best_acc = 0.0

      for i in pbar:
        # 生成一个新样本 (数据+标签)
        flow_type, flow_profile = flow_gen.get_random_flow()
        # 标签：FlowType.value (1,2,3) -> 损失函数 (0,1,2)
        label = flow_type.value - 1 
        label_tensor = torch.tensor([label], dtype=torch.long) #

        input_tensor = send_packet_and_capture(
          server=server,
          client=client,
          flow_type=flow_type,
          duration_sec=FLOW_DURATION,
          n_packets_to_capture=N_PACKETS_TO_CAPTURE
        ).float()

        # if i < 20:
        #   pbar.write(f"[Debug] Flow {i} Type: {flow_type.name}")
        #   pbar.write(f"[Debug] Tensor {i} Shape: {input_tensor.shape}")
        #   pbar.write(f"[Debug] Tensor {i} Data:  {input_tensor}")

        logits = model(input_tensor)            # 向前传播
        loss = criterion(logits, label_tensor)  # 计算损失
        loss = loss/ACCUMULATION_STEPS          # 归一化
        loss.backward()                         # 反向传播，梯度积累
        
        # 记录统计数据
        total_loss += loss.item()*ACCUMULATION_STEPS
        predicted_index = torch.argmax(logits, dim=1).item()

        if predicted_index == label:
          correct_predictions += 1
        
        if (i+1) % ACCUMULATION_STEPS == 0:
          # 更新模型
          optimizer.step()
          optimizer.zero_grad()

          # 更新统计数据
          avg_loss = total_loss / ACCUMULATION_STEPS
          accuracy = correct_predictions / ACCUMULATION_STEPS

          #  更新学习率
          scheduler.step(avg_loss)
          current_lr = optimizer.param_groups[0]['lr']

          # 打印当前训练情况
          pbar.write({f'Loss:{avg_loss:.4f} Acc: {accuracy:.2%} LR: {current_lr:.1e}'})

          # 保存最佳模型
          if best_acc < accuracy:
            torch.save(model.preference_module.state_dict(), Lstm_PATH)
            torch.save(model.state_dict(), Classifier_PATH)
            best_acc = accuracy

          total_loss = 0.0
          correct_predictions = 0

      info(f"====训练完成====\n")
      print(f"best acc: {best_acc}")

  except Exception as e:
    info(f"\n--- 仿真出错 ---")
    info(f"{e}\n")
    import traceback
    traceback.print_exc()


if __name__ == '__main__':
  setLogLevel('info')
  
  if os.getuid() != 0:
    print("错误：此脚本必须以 root 权限 (sudo) 运行。")
  else:
    run_live_training()
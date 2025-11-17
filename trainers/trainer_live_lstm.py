import os
import sys
import networkx as nx
import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mininet.log import setLogLevel, info

# 引入你的自定义模块
# 确保 MininetController.py 里的 normalize_fingerprint 已经修改正确
from MS.Env.MininetController import get_a_mininet, get_a_fingerprint
from MS.Env.FlowGenerator import FlowGenerator 
from MS.Env.TensorLog import append_matrix_to_file
from MS.LSTM.Pretrain.TmpClassifier import RNNClassifier 

# ================= 配置中心 =================
class Config:
  # 训练参数
  LR = 1e-3                # 初始学习率
  BATCH_SIZE = 16          # [关键] 每收集 16 个流更新一次。太大会等很久。
  TOTAL_EPOCHS = 50        # 总轮数 (Live模式下由 Steps 决定实际量)
  STEPS_PER_EPOCH = 10     # 每轮多少个 Batch (10 * 16 = 160 个流/Epoch)
  
  # 路径
  MODEL_DIR = "./trained_model"
  LSTM_PATH = os.path.join(MODEL_DIR, "trained_lstm.pth")
  CLASSIFIER_PATH = os.path.join(MODEL_DIR, "trained_classifier.pth")
  LOG_DIR = "./train-log"
  
  # 模型结构 (3分类任务)
  INPUT_DIM = 2            # (PacketSize, IAT)
  RNN_LAYERS = 2
  NUM_CLASSES = 3
  HIDDEN_DIM = 128
  
  # 采集参数
  N_PACKETS = 30           # 序列长度

CONFIG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 核心函数 =================
def get_model_optimizer():
  """初始化模型和优化器，如果存在旧权重则自动加载 (断点续训)"""
  model = RNNClassifier(
    input_dim=CONFIG.INPUT_DIM, 
    rnn_hidden_dim=CONFIG.HIDDEN_DIM, 
    num_classes=CONFIG.NUM_CLASSES, 
    rnn_layers=CONFIG.RNN_LAYERS
  ).to(device)
  
  optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)
  start_epoch = 1
  best_acc = 0.0

  # --- 断点续训逻辑 ---
  if os.path.exists(CONFIG.CLASSIFIER_PATH):
    print(f"[ms] 发现已训练的模型: {CONFIG.CLASSIFIER_PATH}")
    try:
      checkpoint = torch.load(CONFIG.CLASSIFIER_PATH, map_location=device)
      # 兼容性处理：如果保存的是整个 state_dict
      if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
      else:
        # 如果保存的是整个 model 对象 (旧习惯)
        model = checkpoint
      print(f"[ms] -> 成功加载权重，继续训练...")
    except Exception as e:
      print(f"[ms] -> 加载失败 ({e})，将重新开始训练。")
  else:
      print(f"[ms] 未发现旧模型，从头开始训练。")

  return model, optimizer, start_epoch, best_acc

def collect_batch_safe(net, server, client, flow_gen, batch_size):
  """
  安全地从 Mininet 采集一个 Batch，忽略个别失败的流。
  """
  batch_inputs = []
  batch_labels = []
  
  collected_count = 0
  pbar_str = f"采集 ({collected_count}/{batch_size})"
  
  while collected_count < batch_size:
    try:
      # 1. 随机流
      flow_type, _ = flow_gen.get_random_flow()
      label = flow_type.value - 1
      
      # 2. 抓包 (耗时操作)
      # 注意：MininetController 内部已经做了归一化！
      fingerprint = get_a_fingerprint(
        server=server,
        client=client,
        flow_type=flow_type,
        n_packets_to_capture=CONFIG.N_PACKETS
      ).float()

      # 直接添加到列表，信任 Controller 的输出
      batch_inputs.append(fingerprint)
      batch_labels.append(label)
      collected_count += 1
        
    except Exception as e:
      print(f"\n[Error] 采集单个样本时出错 (已跳过): {e}")
      # 稍微冷却一下 Mininet，防止连续报错
      time.sleep(1)
          
  # 组装 Batch
  input_tensor = torch.cat(batch_inputs, dim=0).to(device)
  label_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)
  
  return input_tensor, label_tensor

# ================= 主循环 =================
def run_live_training():
    # 1. 准备环境
    os.makedirs(CONFIG.MODEL_DIR, exist_ok=True)
    model, optimizer, start_epoch, best_acc = get_model_optimizer()
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 2. 启动 Mininet
    g = nx.Graph()
    g.add_edges_from([("1", "2")]) # 简单两节点拓扑
    
    print(f"[ms] 启动 Mininet 环境...")
    try:
      with get_a_mininet(g) as net:
        net.pingAll()
        h1, h2 = net.get('h1', 'h2')
        flow_gen = FlowGenerator()
        
        print(f"[ms] 环境就绪。开始 Live 训练 (BatchSize={CONFIG.BATCH_SIZE})")
        
        for epoch in range(start_epoch, CONFIG.TOTAL_EPOCHS + 1):
          model.train()
          epoch_loss = 0.0
          epoch_correct = 0
          epoch_samples = 0
          
          # 创建目录
          os.makedirs(f"{CONFIG.LOG_DIR}/Epoch{epoch}", exist_ok=True)
          
          # 进度条
          pbar = tqdm(range(CONFIG.STEPS_PER_EPOCH), desc=f"Epoch {epoch}")
          
          for step in pbar:
            # --- 阶段 A: 采集 (IO瓶颈) ---
            pbar.set_description(f"Epoch {epoch} [采集...]")
            inputs, labels = collect_batch_safe(net, h1, h2, flow_gen, CONFIG.BATCH_SIZE)
            
            # --- 阶段 B: 训练 (GPU瞬间完成) ---
            pbar.set_description(f"Epoch {epoch} [训练...]")
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # --- 统计 ---
            loss_val = loss.item()
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            
            epoch_loss += loss_val
            epoch_correct += correct
            epoch_samples += CONFIG.BATCH_SIZE
            
            # 实时更新进度条
            acc_rate = correct / CONFIG.BATCH_SIZE
            pbar.set_postfix({"Loss": f"{loss_val:.3f}", "Acc": f"{acc_rate:.0%}"})
            
            # 记录第一个样本用于人工检查
            if step == 0:
              append_matrix_to_file(
                inputs[0].cpu(), 
                f"{CONFIG.LOG_DIR}/Epoch{epoch}/sample_batch0.log", 
                epoch
              )

          # --- Epoch 结算 ---
          avg_loss = epoch_loss / CONFIG.STEPS_PER_EPOCH
          avg_acc = epoch_correct / epoch_samples
          
          pbar.write(f"[总结 Epoch {epoch}] Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2%}")
          
          # 无论是否最佳，每轮都保存一次 checkpoint (为了断点续训)
          torch.save(model.state_dict(), CONFIG.CLASSIFIER_PATH)
          
          # 如果是最佳模型，额外保存 LSTM 主体
          if avg_acc >= best_acc:
            best_acc = avg_acc
            pbar.write(f"[ms] >>> 发现新最佳模型 (Acc: {best_acc:.2%})，已保存 LSTM 主体。")
            torch.save(model.preference_module.state_dict(), CONFIG.LSTM_PATH)

    except KeyboardInterrupt:
      print("\n[ms] 用户手动停止训练。模型已保存。")
    except Exception as e:
      print(f"\n[Error] 严重错误: {e}")
      traceback.print_exc()
    finally:
      print("[ms] 清理 Mininet...")
      # with 语句会自动调用 net.stop()，这里是双重保险

if __name__ == '__main__':
    setLogLevel('error')
    if os.getuid() != 0:
      print("【错误】必须使用 sudo 运行此脚本 (Mininet要求)")
    else:
      run_live_training()
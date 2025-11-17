import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# ==============================================================================
# 1. 全局常量与归一化
# ==============================================================================

# 归一化常量 (必须在预训练和 RL 阶段保持一致)
MIN_PACKET_SIZE = 40.0   # 最小包 (e.g., TCP ACK)
MAX_PACKET_SIZE = 1600.0 # 最大包 (e.g., Ethernet MTU)
MIN_IAT = 0.0          
MAX_IAT = 0.200      # 200 毫秒 (假设的最大观测 IAT)

# 标签映射
FLOW_TYPES = {
  "VoIP": 0,
  "Interactive": 1,
  "Streaming": 2,
}

NUM_CLASSES = len(FLOW_TYPES)
N_PACKETS = 30 # 指纹长度 N

def normalize_fingerprint(fp: np.ndarray) -> np.ndarray:
  """
  对 (N, 2) 的指纹 [PacketSize, IAT] 进行 Min-Max 归一化。
  这是一个原地 (in-place) 修改，最后返回裁剪后的数组。
  """
  # 归一化 PacketSize (列 0)
  fp[:, 0] = (fp[:, 0] - MIN_PACKET_SIZE) / (MAX_PACKET_SIZE - MIN_PACKET_SIZE)
  # 归一化 IAT (列 1)
  fp[:, 1] = (fp[:, 1] - MIN_IAT) / (MAX_IAT - MIN_IAT)
  
  # 裁剪到 [0, 1] 范围，防止因模拟噪声导致超出边界
  return np.clip(fp, 0.0, 1.0) 

# ==============================================================================
# 2. 完整的 FlowGenerator 类
# ==============================================================================

class FlowGenerator:
  """
  动态模拟生成 (N, 2) 的指纹和标签
  用于阶段 1.A (监督预训练) 和 阶段 2 (RL 状态生成)
  """
  def __init__(self, num_packets: int = N_PACKETS):
    """
    初始化生成器
    :param num_packets: 指纹长度 (N)
    """
    self.N = num_packets
    # 使用 NumPy 1.17+ 推荐的现代随机数生成器
    self.rng = np.random.default_rng()

  def _generate_voip(self):
    """
    模拟 VoIP (QoE: G.114 / G.107, 实时会话流)
    特点: 恒定小包, 恒定 IAT (e.g., 20ms)
    """
    # G.711 负载 (160) + RTP/UDP/IP (40)
    size = self.rng.normal(loc=200.0, scale=10.0, size=(self.N, 1))
    # 20ms 采样间隔，带轻微抖动 (jitter)
    iat = self.rng.normal(loc=0.020, scale=0.002, size=(self.N, 1)) 
    return np.hstack([size, iat]), FLOW_TYPES["VoIP"]

  def _generate_interactive(self):
    """
    模拟 Interactive (QoE: 3GPP QCI 80, 游戏/交互)
    特点: 突发小包, 较短 IAT (e.g., 30-50ms)
    """
    # 模拟客户端 -> 服务器 (小包), 服务器 -> 客户端 (中包)
    # 假设 60% 是小的上行包，40% 是稍大的下行包
    size = self.rng.choice([60.0, 150.0], p=[0.6, 0.4], size=(self.N, 1))
    # 增加少量噪声
    size += self.rng.normal(loc=0.0, scale=5.0, size=(self.N, 1))
    
    # 30ms 游戏 tick 速率，但有突发
    iat = self.rng.normal(loc=0.030, scale=0.005, size=(self.N, 1))
    
    # 模拟突发（非常小的 IAT），约 20% 的包是突发的
    burst_indices = self.rng.choice(self.N, size=int(self.N * 0.2), replace=False)
    iat[burst_indices] = self.rng.uniform(0.001, 0.005, size=(int(self.N * 0.2), 1))
    
    return np.hstack([size, iat]), FLOW_TYPES["Interactive"]

  def _generate_streaming(self):
    """
    模拟 Streaming (QoE: 3GPP QCI 6, 流媒体)
    特点: 突发大包 (TCP), IAT 呈双峰分布 (突发期 vs 缓冲期)
    """
    # 主要是 MTU 大小的包
    size = self.rng.normal(loc=1460.0, scale=50.0, size=(self.N, 1))
    # 模拟 10% 的 TCP ACK 小包
    ack_indices = self.rng.choice(self.N, size=int(self.N * 0.1), replace=False)
    size[ack_indices] = self.rng.normal(loc=60.0, scale=5.0, size=(int(self.N * 0.1), 1))
    
    # IAT: 突发期 (包间隙小) vs. 缓冲期 (包间隙大)
    iat_burst = self.rng.uniform(0.0001, 0.002, size=(self.N, 1))
    iat_buffer = self.rng.uniform(0.050, 0.150, size=(self.N, 1))
    
    # 假设 70% 的包处于突发期 (正在下载一个片段)
    mask = self.rng.random(size=(self.N, 1)) < 0.7
    iat = np.where(mask, iat_burst, iat_buffer)
    
    return np.hstack([size, iat]), FLOW_TYPES["Streaming"]

  def generate_sample(self):
    """
    随机选择一个类型并生成归一化的指纹
    """
    # 随机选择一个流类型 (0, 1, 2)
    flow_type_idx = self.rng.integers(0, NUM_CLASSES)
    
    if flow_type_idx == FLOW_TYPES["VoIP"]:
      fp, label = self._generate_voip()
    elif flow_type_idx == FLOW_TYPES["Interactive"]:
        fp, label = self._generate_interactive()
    elif flow_type_idx == FLOW_TYPES["Streaming"]:
        fp, label = self._generate_streaming()
        
    # 关键：归一化
    fp_normalized = normalize_fingerprint(fp)
    
    # 转换为 PyTorch Tensors
    fp_tensor = torch.tensor(fp_normalized, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.long)
    
    return fp_tensor, label_tensor

class FlowFingerprintDataset(Dataset):
  """用于动态生成数据的 PyTorch Dataset"""
  def __init__(self, generator: FlowGenerator, epoch_size: int = 20000):
    self.generator = generator
    self.epoch_size = epoch_size

  def __len__(self):
    return self.epoch_size

  def __getitem__(self, idx):
    # 每次调用都生成新样本，确保训练的泛化性
    return self.generator.generate_sample()
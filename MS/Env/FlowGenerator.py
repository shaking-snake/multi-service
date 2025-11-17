from enum import Enum
import random

class FlowType(Enum):
  VOIP = 1      # 实时会话
  STREAMING = 2 # 流媒体
  GAMING = 3    # 游戏

# 存储每种流类型的 iperf 参数和 QoE 严格要求 (用于 Reward 函数)
FLOW_PROFILES = {
  FlowType.VOIP: {
    'protocol': 'UDP',       # 改名为 protocol 更通用
    'ditg_preset': 'VoIP -x G.711.2' # D-ITG 专用参数
    'qoe_critical': {'max_delay': 150, 'max_jitter': 50}, # ms
    'reward_fn': 'E-Model'
  },
  FlowType.STREAMING: {
    'protocol': 'UDP',
    'ditg_preset': 'CSa'
    'qoe_critical': {'min_bandwidth': 5, 'max_loss_rate': 1e-6}, # Mbps, %
    'reward_fn': '3GPP-QCI6'
  },
  FlowType.INTERACTIVE: {
    'protocol': 'TCP',
    'ditg_manual': '-B E 2000 E 3000 -c 1460 -C 1000'    # 视频流手动参数
    'qoe_critical': {'max_delay': 50, 'max_jitter': 30}, # ms
    'reward_fn': '3GPP-QCI80'
  }
}

class FlowGenerator:
  def get_random_flow(self) -> tuple[FlowType, dict]:
    """随机选择一个流类型及其配置文件。"""
    flow_type = random.choice(list(FlowType))
    profile = FLOW_PROFILES[flow_type]
    return flow_type, profile

# --- 示例用法 ---
# flow_gen = FlowGenerator()
# current_type, current_profile = flow_gen.get_random_flow()
# print(f"当前流类型: {current_type.name}")
# print(f"iperf模式: {current_profile['iperf_mode']}, 目标速率: {current_profile['target_rate']}")
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class SO100ReachEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("so100.xml")
        self.data = mujoco.MjData(self.model)
        
        # 动作空间：6个关节目标位置
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # 观察空间：关节位置 + 末端位置
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        self.target = np.array([0.3, 0.0, 0.2])
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # 重置关节
        self.data.qpos[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
    
    def step(self, action):
        # 执行动作
        self.data.ctrl[:] = action
        
        # 仿真10步
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # 计算奖励
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = reward > -0.01  # 到达目标
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        joint_pos = self.data.qpos[:6].copy()
        end_effector = self.data.xpos[self.model.nbody-1].copy()
        return np.concatenate([joint_pos, end_effector]).astype(np.float32)
    
    def _compute_reward(self):
        end_effector = self.data.xpos[self.model.nbody-1]
        distance = np.linalg.norm(end_effector - self.target)
        return -distance  # 越近越好

# 注册
gym.register(id="SO100-Reach-v0", entry_point="SO100ReachEnv")
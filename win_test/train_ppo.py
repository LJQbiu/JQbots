import gymnasium as gym
from stable_baselines3 import PPO
from gem_register import SO100ReachEnv

gym.register(id="SO100-Reach-v0", entry_point="SO100ReachEnv")

# 创建环境
env = gym.make("SO100-Reach-v0")

# 训练 PPO
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=100000)

# 保存
model.save("so100_ppo_reach")

# 测试
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path(r"win_test\SO101\so101_new_calib.xml")
data = mujoco.MjData(model)

# 获取末端执行器位置（假设最后一个 body 是末端）
def get_end_effector_pos():
    # body id 通常是最后一个
    body_id = model.nbody - 1
    return data.xpos[body_id].copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 目标位置（在机械臂前方）
    target = np.array([-1, -1, -1])
    
    # 简单的 PD 控制
    for _ in range(2000):
        # 当前末端位置
        current = get_end_effector_pos()
        
        # 计算误差
        error = target - current
        
        # 简单的启发式：根据误差调整关节
        # 实际应该用 IK，这里先用简单策略
        data.ctrl[0] = 0.5 * error[0]  # shoulder_pan
        data.ctrl[1] = 0.5 * error[2]  # shoulder_lift (z方向)
        
        # 步进
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
    
    print("Final position:", get_end_effector_pos())
    print("Target:", target)
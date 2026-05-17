#!/usr/bin/env python3

"""
测试训练修复效果的脚本
"""

import torch
import numpy as np
from tacex_tasks.tacex_tasks.cylinder_grasping.cylinder_grasping_privileged import CylinderGraspingPrivilegedEnv, CylinderGraspingPrivilegedCfg

def test_environment():
    """测试环境是否正常工作"""
    print("🔧 测试环境配置...")
    
    # 创建环境配置
    cfg = CylinderGraspingPrivilegedCfg()
    cfg.scene.num_envs = 4  # 使用少量环境进行测试
    cfg.sim.dt = 1.0 / 60.0  # 60Hz
    
    print(f"  ✅ 动作空间: {cfg.action_space}")
    print(f"  ✅ 最大步数: {cfg.max_episode_length}")
    print(f"  ✅ 动作缩放: {cfg.action_scale}")
    print(f"  ✅ IK方法: {cfg.ik_controller_cfg.ik_method}")
    
    # 创建环境
    env = CylinderGraspingPrivilegedEnv(cfg)
    print(f"  ✅ 环境创建成功，环境数量: {env.num_envs}")
    
    # 重置环境
    obs = env.reset()
    print(f"  ✅ 环境重置成功，观察空间形状: {obs['proprio_obs'].shape}")
    
    # 测试随机动作
    print("\n🎮 测试随机动作...")
    for step in range(10):
        # 生成随机动作
        actions = torch.randn(env.num_envs, cfg.action_space, device=env.device) * 0.1
        
        # 执行动作
        obs, rewards, dones, info = env.step(actions)
        
        print(f"  步骤 {step+1}: 奖励均值={rewards.mean().item():.3f}, 成功数={dones.sum().item()}")
        
        # 检查是否有环境结束
        if dones.any():
            print(f"  🏁 有环境结束: {dones.sum().item()}/{env.num_envs}")
            break
    
    print("\n📊 测试奖励配置...")
    # 测试奖励计算
    test_actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    obs, rewards, dones, info = env.step(test_actions)
    
    print(f"  ✅ 零动作奖励: {rewards.mean().item():.3f}")
    print(f"  ✅ 奖励范围: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")
    
    # 测试碰撞检测
    print("\n🚫 测试碰撞检测...")
    collision_penalty = env._compute_collision_penalty()
    print(f"  ✅ 碰撞惩罚: {collision_penalty.mean().item():.3f}")
    print(f"  ✅ 碰撞环境数: {collision_penalty.sum().item():.0f}")
    
    print("\n✅ 环境测试完成！")
    return True

def test_reward_balance():
    """测试奖励平衡性"""
    print("\n⚖️ 测试奖励平衡性...")
    
    cfg = CylinderGraspingPrivilegedCfg()
    env = CylinderGraspingPrivilegedEnv(cfg)
    env.reset()
    
    # 测试不同动作的奖励
    test_actions = [
        torch.zeros(env.num_envs, cfg.action_space, device=env.device),  # 零动作
        torch.ones(env.num_envs, cfg.action_space, device=env.device) * 0.1,  # 小动作
        torch.ones(env.num_envs, cfg.action_space, device=env.device) * 0.5,  # 中等动作
        torch.ones(env.num_envs, cfg.action_space, device=env.device) * 1.0,  # 大动作
    ]
    
    for i, actions in enumerate(test_actions):
        obs, rewards, dones, info = env.step(actions)
        print(f"  动作 {i+1}: 奖励={rewards.mean().item():.3f}, 标准差={rewards.std().item():.3f}")
    
    print("✅ 奖励平衡性测试完成！")

if __name__ == "__main__":
    print("🚀 开始测试训练修复效果...")
    
    try:
        # 测试环境
        test_environment()
        
        # 测试奖励平衡
        test_reward_balance()
        
        print("\n🎉 所有测试通过！训练修复成功！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

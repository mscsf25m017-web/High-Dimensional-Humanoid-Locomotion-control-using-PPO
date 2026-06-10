# High-Dimensional Humanoid Locomotion Control using Proximal Policy Optimization (PPO)

> Deep Reinforcement Learning for Stable 3D Humanoid Walking in MuJoCo Humanoid-v4

---

## Overview

This project implements a scalable, memory-efficient PPO training pipeline for a high-dimensional 3D humanoid agent operating in the **Gymnasium Humanoid-v4** environment. The agent controls **17 rotational degrees of freedom** and must learn coordinated, energy-efficient bipedal locomotion entirely from reward signals — no demonstrations, no motion capture data.

Key contributions:
- Parallelized training across **12 MuJoCo simulation workers**
- Observation and reward normalization for stable gradient flow
- Composite reward shaping balancing forward speed, posture health, and actuator efficiency
- Trained to **15 million timesteps** on CPU with consistent convergence

---

## Demo

> 🎥 https://drive.google.com/drive/folders/1cIoM3F1bo2BtGCHtBAxydhS0tS103tVx?usp=share_link

---

## Environment

**Humanoid-v4** (Gymnasium + MuJoCo) is one of the most demanding continuous-control benchmarks, requiring simultaneous coordination of 17 joints across the full body.

| Property | Value |
|---|---|
| Simulator | MuJoCo |
| Observation Space | Continuous, high-dimensional |
| Action Space | Continuous torque control |
| Control DoF | 17 rotational joints |
| Task | Stable forward locomotion |

### Joint Configuration

| Body Segment | Joints | Role |
|---|---|---|
| Abdomen | 3 | Torso stabilization and posture |
| Hips | 6 | Stride generation |
| Knees | 2 | Leg extension and ground support |
| Ankles | 4 | Foot placement and balance |
| Shoulders | 4 | Arm swing coordination |
| Elbows | 2 | Arm articulation |

---

## Reward Function

The agent optimizes a composite reward at each timestep:

```
R_total = R_forward + R_healthy − C_control − C_impact
```

| Component | Description |
|---|---|
| `R_forward` | Reward for forward velocity along the x-axis |
| `R_healthy` | Survival bonus for maintaining valid upright posture |
| `C_control` | Penalty on actuator torque magnitude (encourages smooth motion) |
| `C_impact` | Penalty on large ground-contact forces (discourages unstable footfalls) |

---

## Method: PPO (Actor-Critic)

PPO was chosen for its well-established stability in high-dimensional continuous-control settings. The clipped surrogate objective prevents destructive policy updates while maintaining strong sample efficiency.

**Policy:** MLP (Multi-Layer Perceptron) via Stable-Baselines3  
**Backend:** PyTorch  
**Device:** CPU

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | `1e-4` |
| `n_steps` | `1024` |
| Batch Size | `256` |
| Epochs per Update | `10` |
| Discount Factor (γ) | `0.99` |
| GAE Lambda (λ) | `0.95` |
| Clip Range (ε) | `0.10` |
| Entropy Coefficient | `0.00` |
| Value Function Coefficient | `0.50` |
| Max Gradient Norm | `0.50` |
| Parallel Environments | `12` |
| Total Timesteps | `15,000,000` |

---

## Training Pipeline

```
1. Initialize 12 parallel Humanoid-v4 environments (VecEnv)
2. Wrap with VecNormalize (observation + reward normalization)
3. Collect rollouts from all workers (n_steps × n_envs per update)
4. Estimate advantages via Generalized Advantage Estimation (GAE)
5. Update Actor and Critic networks using clipped PPO objective
6. Save checkpoints at regular intervals
7. Evaluate and record final policy
```

---

## Results

After **15 million timesteps** of training, the agent achieved the following behaviors:

| Metric | Result |
|---|---|
| Stable upright posture | ✅ Achieved |
| Continuous forward locomotion | ✅ Achieved |
| Coordinated lower-body movement | ✅ Achieved |
| Balance recovery | ✅ Improved |
| Smooth walking trajectory | ✅ Natural gait observed |
| Reduced actuator effort | ✅ Control cost minimized |

The final policy maintained stable forward movement throughout full episodes without collapsing, representing a significant improvement over the random baseline.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
gymnasium[mujoco]
stable-baselines3
torch
numpy
tensorboard
```

---

## Usage

### Train from scratch

```bash
python train.py
```

### Evaluate a saved model

```bash
python evaluate.py --model_path models/ppo_humanoid_final.zip
```

### Record a demo video

```bash
python record_video.py --model_path models/ppo_humanoid_final.zip
```

### Monitor training with TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Project Structure

```
├── train.py               # Main training script
├── evaluate.py            # Policy evaluation
├── record_video.py        # Demo video generation
├── models/                # Saved model checkpoints
├── logs/                  # TensorBoard logs
├── videos/                # Recorded episode videos
└── requirements.txt
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Gymnasium | Environment interface |
| MuJoCo | Physics simulation |
| Stable-Baselines3 | PPO implementation |
| PyTorch | Neural network backend |
| NumPy | Numerical operations |
| TensorBoard | Training visualization |

---

## Demonstration of Project
> 🎥https://drive.google.com/file/d/14hhdwY8Cg-0D7nztTv6R6pLJEQe8MfBO/view?usp=share_link

## Author

**Rida Saeed**  
M.Sc. Computer Science — Roll No. MSCSF25M017  
Punjab University College of Information Technology (PUCIT), Lahore

**Supervisor:** Dr. Nazar Ameen Khan  
Assistant Professor, PUCIT, University of the Punjab, Lahore, Pakistan

# RL Agent on Classic Control Task

## Overview
Train a PPO agent on CartPole or LunarLander using Stable-Baselines3. Record a video of the agent converging, and deploy an interactive inference demo where users can watch the trained agent play.

## Category
Machine Learning / Reinforcement Learning

## Stack
- **Stable-Baselines3** — PPO implementation
- **Gymnasium** — CartPole-v1 / LunarLander-v2 environments
- **Matplotlib** — reward curve + training plots
- **Gradio or Streamlit** — interactive inference demo
- **OpenCV / imageio** — video recording

## Key Features
- Training script with logging (reward, episode length)
- Reward curve plot showing convergence
- Recorded video of trained agent performing the task
- Interactive demo: user triggers an episode and watches the agent in real-time (rendered frames streamed to UI)
- Optionally: side-by-side random agent vs. trained agent

## Architecture
1. Train PPO with SB3, save checkpoints
2. Log rewards per episode → plot convergence curve
3. Roll out trained policy → capture frames → encode as GIF/MP4
4. Gradio app: load model, run episode, display video

## Tasks
- **CartPole-v1** — balancing pole, fast to train (~50k steps)
- **LunarLander-v2** — more visually impressive, ~500k steps

## Portfolio Value
- Demonstrates RL foundations beyond just supervised learning
- Reward convergence video is compelling evidence of training
- Interactive demo makes it accessible to non-technical reviewers

## Milestones
- [ ] Training script with SB3 PPO
- [ ] Reward curve logging + plot
- [ ] Video recording of trained agent
- [ ] Gradio inference demo
- [ ] Deploy on HuggingFace Spaces
- [ ] README with training details + demo video

## Notes
<!-- Add implementation notes, decisions, and progress here -->

# Learning to Run: Reinforcement Learning for Bipedal Locomotion


Task was to train a deep reinforcement learning agent to control a bipedal robot in the `rldurham/Walker` Box2D environment. The agent was trained in both the standard/easy environment and the more challenging hardcore environment, where the robot must navigate obstacles such as stumps, ladders and pitfalls.

## Project Overview

The aim of the project was to develop a sample-efficient reinforcement learning agent that could learn to walk effectively within the required episode limits:

- Easy environment: up to 1000 training episodes
- Hardcore environment: up to 2000 training episodes

My final agent uses **Truncated Quantile Critics (TQC)**, an entropy-regularised distributional actor-critic method based on Soft Actor-Critic. Instead of estimating a single Q-value, the critic estimates a distribution over returns using quantiles, with the highest quantiles truncated to reduce overestimation bias.

The final submitted report is titled:

**Learning to Walk Randomly**

## Repository Contents

```text
.
├── feedback_hqcb32.pdf
├── hqcb32-agent-code.ipynb
├── hqcb32-agent-code-hardcore.ipynb
├── hqcb32-agent-log.txt
├── hqcb32-agent-log-hardcore.txt
├── hqcb32-agent-paper.pdf
├── hqcb32-agent-video,...mp4
├── hqcb32-agent-video-hardcore,...mp4
└── README.md
```

## Files

### `hqcb32-agent-paper.pdf`

The final scientific report describing the methodology, architecture, experiments, convergence results, limitations and future work.

### `hqcb32-agent-code.ipynb`

Notebook containing the implementation and training code for the easy `rldurham/Walker` environment.

### `hqcb32-agent-code-hardcore.ipynb`

Notebook containing the implementation and training code for the hardcore version of the `rldurham/Walker` environment.

### `hqcb32-agent-log.txt`

Training log for the easy environment.

### `hqcb32-agent-log-hardcore.txt`

Training log for the hardcore environment.

### `hqcb32-agent-video,...mp4`

Video recording of the best-performing easy-environment agent.

### `hqcb32-agent-video-hardcore,...mp4`

Video recording of the best-performing hardcore-environment agent.


## Method Summary

The agent is based on **Truncated Quantile Critics (TQC)**. The key components are:

- Stochastic Gaussian actor with `tanh` squashing for continuous actions
- Five quantile critics
- Twenty-five quantiles per critic
- Truncation of the most optimistic quantiles to reduce Q-value overestimation
- Observation normalisation using running mean and variance
- LayerNorm in the critic networks
- Replay buffer with off-policy updates
- Polyak averaging for target network updates
- Entropy-regularised learning inspired by Soft Actor-Critic

The easy and hardcore environments used slightly different configurations. The easy agent used a hidden width of 256 and 5,000 warmup steps, while the hardcore agent used a hidden width of 400 and 10,000 warmup steps. The hardcore configuration also used entropy annealing, median critic aggregation and reward scaling.

## Results

The final agent achieved:

| Environment | Best Episode Return |
|---|---:|
| Easy | 244.2 |
| Hardcore | 224.6 |

The easy agent showed strong convergence, with the rolling mean surpassing 240 by around episode 386. The hardcore agent learned more slowly and remained less stable, but still reached competent walking behaviour, with a best return of 224.6.

## Main Findings

Observation normalisation and critic LayerNorm produced the clearest improvement in performance. Prioritised experience replay improved early learning but did not raise the final performance plateau. Entropy annealing helped in the hardcore environment but worsened late-stage collapse in the easy environment.

The easy environment showed strong final performance but still had occasional late-stage instability. The hardcore environment showed partial convergence, with good performance on many terrain layouts but continued sensitivity to difficult procedurally generated obstacles.

## Limitations

The main limitation in the easy environment was late-stage instability after the agent had already learned a strong gait. This may have been caused by residual exploration from the entropy term or shifting observation normalisation statistics.

The main limitation in the hardcore environment was lack of robustness across all terrain layouts. Although the agent often performed well, some episodes still failed badly, suggesting that the policy had not fully generalised to every obstacle configuration.

## Future Work

Possible extensions include:

- Adaptive entropy tuning to reduce exploration once the policy stabilises
- Longer hardcore training beyond 2000 episodes
- Adding short-term observation history using an LSTM or Transformer encoder
- Further investigation into observation normalisation freezing late in training
- More robust exploration strategies for difficult terrain layouts

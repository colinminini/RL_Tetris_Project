# RL Tetris Project

Train and evaluate reinforcement learning agents for Tetris using
`tetris-gymnasium`. The repo includes:
- DQN after-state agent with Dellacherie features (holes, bumpiness, aggregate
  height, lines cleared).
- Vanilla policy gradient CNN agent on the raw board state.
- Baseline heuristics (greedy, random, hard drop) and a manual-play script.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `pip` errors on `cv2`, remove that line and keep `opencv-python` installed
(`cv2` comes from `opencv-python`).

## Train

Policy gradient CNN:

```bash
python train_pg_cnn.py --episodes 500 --normalize-returns --save-path checkpoints/pg_cnn.pt
```

DQN after-state (feature-based):

```bash
python train_dqn_afterstate.py --episodes 1000 --save-path checkpoints/dqn_afterstate.pt
```

Use `--render` to watch training (slow).

## Evaluate

```bash
python evaluate_pg_cnn.py --model-path checkpoints/pg_cnn.pt --episodes 20
python evaluate_dqn_afterstate.py --model-path checkpoints/dqn_afterstate.pt --episodes 20
```

Add `--render` to visualize gameplay. For the policy gradient agent, pass
`--stochastic` to sample actions instead of greedy play.

## Baselines and Manual Play

```bash
python Baseline/view_episode_policy_greedy.py
python Baseline/view_episode_policy_random.py
python Baseline/view_episode_policy_down.py
python Baseline/evaluate_policy_greedy.py
python Baseline/play_tetris.py
```

## Notes

- Checkpoints in `checkpoints/` are pre-trained weights. The training and
  evaluation scripts default to `tetris_code/checkpoints/`; override with
  `--save-path` and `--model-path` to use the existing `checkpoints/` folder.
- `run_scripts.ipynb` shows an end-to-end workflow; update the paths if you run
  it from this repo root.

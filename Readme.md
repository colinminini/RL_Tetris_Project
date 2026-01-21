# RL Tetris Project

Train and evaluate reinforcement learning agents for Tetris using
`tetris-gymnasium`. The repo includes:
- DQN after-state agent with Dellacherie features (holes, bumpiness, aggregate
  height, lines cleared).
- Baseline heuristics (greedy, random, hard drop) and a manual-play script.

## Environment source and scoring

This project uses the Gymnasium-compatible `tetris-gymnasium` environment
(`gym.make("tetris_gymnasium/Tetris")`). The score reported in baseline scripts
and evaluation is the environment return:

```text
env_return = sum_t env_reward_t
```

Where `env_reward_t` is the raw reward returned by `env.step(...)` at each step
(or macro-action sequence in the after-state agent). The exact reward scheme is
defined inside `tetris-gymnasium`:

```text
env_reward_t = 0.001  (alive bonus)
             + 1.0 * lines_cleared_t
             - 2.0 * [game_over_t]
```

The reported score/return in the scripts is the sum of these per-step rewards
over an episode.

The default `ActionsMapping` in `tetris-gymnasium` is:

```text
move_left: 0
move_right: 1
move_down: 2
rotate_clockwise: 3
rotate_counterclockwise: 4
hard_drop: 5
swap: 6
no_operation: 7
```

## Greedy baseline policy

The greedy baseline in `tetris_code/policies.py` scores many candidate
placements and picks the one with the best heuristic score.

### Micro-actions, sequences, and macro-actions

- A micro-action is one primitive environment action (see ActionsMapping above).
- A sequence of actions is a short list of micro-actions.
- A macro-action is a sequence that represents "place the current piece"; it represents the possible legal moves from state s.
  (e.g., rotate, move left/right several times, then hard drop).

In the greedy policy, we enumerate sequences that look like:
rotate (0-3 times) -> shift (0..k times) -> hard drop.

### Score definition

For each macro-action from state s, we simulate it on a deep-copied environment and score the
resulting board. The score in code is:

```text
score = min(heights(board)) - 100 * holes(board)
```

Where:

- `heights(board)` returns the index of the first filled cell from the top for
  each playable column (higher values imply lower stacks).
- `holes(board)` counts empty cells that have a filled cell above them.

The large penalty on holes dominates: fewer holes is much more important than
small changes in height.

### How the best action is selected

1. Enumerate all candidate sequences (macro-actions).
2. For each sequence, step through it in a copied env and compute the score.
3. Select the sequence with the maximum score.
4. Execute only the first micro-action of that best sequence in the real env,
   then repeat the process at the next time step.

### How legality is handled

The policy does not implement explicit collision or legality checks. Instead,
it relies on the environment's `step` function during simulation to enforce
legal moves. Any illegal or no-op actions are handled by the env dynamics, and
the resulting board is still scored.

## Tetris state space (order-of-magnitude)

The Tetris state space is enormous. A conservative lower bound is the number of
binary board configurations for a 20x10 grid:

```text
2^(200) ~= 1.6e60
```

The true state space is much larger because it also depends on the current
piece, its orientation, its position, the queue of upcoming pieces, and other
environment details.

## DQN After-State DDQN Details

This repo uses a Double DQN (DDQN) with after-states and macro-actions. The design choices below are tailored for Tetris and for data efficiency.

### After-states and macro-actions

- A macro-action is a short sequence of primitive actions (rotate, shift,
  hard drop) that results in placing the current piece. We enumerate many such
  sequences and treat each as one decision.
- An after-state is the board immediately after a macro-action completes. This
  reduces the action space complexity because we evaluate the resulting board
  states directly.
- The network estimates a value for the after-state, which is combined with the
  immediate shaped reward to choose the best candidate:

```text
score(candidate) = shaped_reward(candidate) + gamma * V(after_state)
```

### Features: what we use and why

We use the Dellacherie feature set, a compact, hand-crafted summary that is
known to correlate well with Tetris performance:

- holes: empty cells with at least one filled cell above in the same column.
- bumpiness: sum of absolute height differences between adjacent columns.
- aggregate height: sum of all column heights.
- lines cleared: number of lines cleared by the last placement.

Why these features:

- They capture key board quality factors (holes and bumpiness are strongly
  predictive of future failure).
- They are low-dimensional and data efficient, enabling fast learning with a
  small MLP instead of a much larger CNN.
- They reflect classic Tetris heuristics and reduce the need for very large
  datasets or long training runs.

#### Who is Dellacherie and why these features

The features are commonly attributed to Pierre Dellacherie, a French Tetris
player who described a strong hand-crafted evaluation function for Tetris. The
feature set is often called the "Dellacherie features" in the Tetris AI
community, and the author of the feature design is Dellacherie himself. These
features are favored because they encode board quality signals that directly
relate to survival and line-clearing potential.

### Reward shaping

Training uses shaped rewards to provide denser feedback than the raw env score.
The shaping encourages line clears and smooth stacks while penalizing bad
placements:

```text
line_reward: {1: 1, 2: 3, 3: 6, 4: 12}
shaped_reward = line_reward - 2.0 * new_holes - 0.5 * bump_increase + 0.1
```

This makes learning more stable by signaling progress even when full lines are
not cleared every move.

### Scoring metric in evaluation

Evaluation reports the environment return (not the shaped reward). The printed
"Env return" is the running sum of raw env rewards for a single episode, and
the final "Average env return" is the mean over all evaluation episodes:

```text
env_return = sum_t env_reward_t
```

### Replay buffer and training loop

We store transitions in a replay buffer to break correlations between sequential
decisions and to reuse experience:

- state_features: after-state feature vector for the chosen macro-action.
- reward: shaped reward for that placement.
- done: terminal flag.
- next_features: features for all next candidate after-states.

Training flow:

- Enumerate candidate macro-actions and their after-state features.
- Select a macro-action using epsilon-greedy over the DDQN score.
- Execute the macro-action, compute shaped reward, and push to the buffer.
- Sample random mini-batches and update the policy network.
- Periodically sync the target network from the policy network.

### Loss function (DDQN target)

We use a DDQN target to reduce overestimation bias:

```text
if done or no next candidates:
    target = reward
else:
    a* = argmax_a Q_policy(next_after_state_a)
    target = reward + gamma * Q_target(next_after_state_a*)
loss = MSE(Q_policy(current_after_state), target)
```

### Why Double DQN and a target network

- The max operator in Q-learning can overestimate values. DDQN reduces this by
  selecting actions with the policy network and evaluating them with a separate
  target network.
- The target network changes slowly (synced periodically), which stabilizes
  learning and avoids chasing a moving target at every update.

### Why DDQN is a good fit for Tetris

- Tetris has high branching and delayed consequences; value-based learning with
  a replay buffer is effective and sample efficient.
- After-states simplify the decision by evaluating final placements rather than
  every low-level step.
- Reward shaping provides dense feedback, which helps in a sparse-reward game.

### Why features + MLP instead of raw grid + CNN

- The feature vector is small and encodes expert knowledge, reducing the amount
  of data needed to learn a good policy.
- A small MLP trains quickly and is easier to stabilize than a deep CNN.
- The raw grid has large spatial redundancy; for this project, features are a
  strong bias that speeds learning.

### Why later episodes take longer

- Early episodes often end quickly and may skip training updates until the
  replay buffer is warm. Later episodes include full backprop updates each step.
- As the policy improves, episodes last longer (more macro-actions per episode),
  which increases total compute time per episode.

### Epsilon (exploration)

Epsilon is the probability of taking a random macro-action instead of the
greedy one. We decay it over time:

```text
epsilon(step) = end + (start - end) * exp(-step / decay)
```

High epsilon early encourages exploration; lower epsilon later favors
exploitation of the learned policy.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `pip` errors on `cv2`, remove that line and keep `opencv-python` installed
(`cv2` comes from `opencv-python`).

## Train

DQN after-state (feature-based):

```bash
python train_dqn_afterstate.py --episodes 1000 --save-path checkpoints/dqn_afterstate.pt
```

Use `--render` to watch training (slow).

## Evaluate

```bash
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
  evaluation scripts default to `DQN_scripts/checkpoints/`; override with
  `--save-path` and `--model-path` to use the existing `checkpoints/` folder.
- `Tetris_RL_script_runner.ipynb` shows an end-to-end workflow.

## Results

- After 100 training episodes, the agent clears ~4k lines on average over a
  10-episode evaluation.
- After 200 training episodes (about 15 minutes on an M4 chip), the agent clears
  ~20k lines on average over 10 episodes with a standard deviation of ~7k.
- This is about 50x higher than the greedy baseline.
- More compute improves performance; as the policy improves, evaluation runs
  take longer (around 35 minutes after the 200-episode training run).

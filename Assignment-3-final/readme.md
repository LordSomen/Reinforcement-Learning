## Assignment Overview

This project implements and analyzes two major families of reinforcement learning algorithms , as mentioned in the Assignment.

1. **Problem 1: Deep Q-Network (DQN)**

   * **Environments:** `MountainCar-v0` and `ALE/Pong-v5`.
   * **Core Concepts:** Replay Buffer, Target Network, $\epsilon$-greedy exploration, and using CNNs for visual data (Pong).
2. **Problem 3: Policy Gradient (REINFORCE with Baseline)**

   * **Environments:** `CartPole-v1` and `LunarLander-v3`.
   * **Core Concepts:** An **Actor-Critic w.r.t State Dependent expected return** model is used.
     * **Actor (`PolicyNetwork`):** Decides which action to take.
     * **Critic (`ValueNetwork`):** Acts as a baseline $V(s)$ to estimate the "advantage" of an action, which stabilizes training.
     * **Techniques:** Reward-to-Go (R2G) and Advantage Normalization.

### Environment Naming

Note: In The Assignment it is mentined older `gym` environment names. But as we can use latest env available as mentioned in the assignment , This project uses the modern `gymnasium` library, which has updated names:

* `MountainCar-v0` $\rightarrow$ `MountainCar-v0` (remains the same)
* `Pong-v0` $\rightarrow$ `ALE/Pong-v5`
* `Cartpole-v0` $\rightarrow$ `CartPole-v1`
* `Lunarlander-v2` $\rightarrow$ `LunarLander-v3`

---

## ⚙️ Environment Setup

This project uses `conda` to manage the environment.

1. **Create the conda environment:**
   ```bash
   conda create --name rl_assignment3 python=3.10
   ```
2. **Activate the environment:**
   ```bash
   conda activate rl_assignment3
   ```
3. **Install PyTorch: **
4. ```bash
   conda install pytorch numpy matplotlib -c pytorch
   ```
5. **Install Gymnasium and dependencies:**
   This installs the "classic control", "box2d" (for LunarLander), and "atari" (for Pong) extras.
   ```bash
   pip install "gymnasium[box2d,atari]"
   ```
6. **Install other required libraries:**
   ```bash
   pip install opencv-python matplotlib
   ```

### Troubleshooting Setup

If you encounter errors during installation, run the following:

* **For `box2d-py` (LunarLander) `swig` error:**

  ```bash
  conda install -c conda-forge swig
  pip install "gymnasium[box2d]"
  ```
* **For `ALE/Pong-v5` (Pong) `Namespace ALE not found` error:**
  You must install `ale-py` and import the ROMs.

  ```bash
  pip install ale-py
  ```

  You also need to `import ale_py` at the top of your Pong script.

#### I have also added the conda env file that I am using for this project , that can also be used to create an environment .

---

## 🗂️ FILE STRUCTURE

This project is organized into scripts, each handling a specific part of the assignment.

TO RUN THE SCRIPTS YOU NEED TO NAVIGATE TO IT'S RESPECTIVE FOLDERS

* `mountain_car`: Main folder for MountainCar-v0, including checkpointing and plotting from the checkpoint.
* `mountain_car/problem-1c`: Main folder for problem 1-c (MountainCar-v0), due to computational constraint I have done only for MountainCar-v0.
* `mountain_car_script.py`: Main training script for MountainCar-v0 .
* `mountain_car_script_hyperparameter_tuning.py`: Script for Problem 1(c) to run multiple experiments (on `learning_rate`) and plot a comparison.
* `pong`: Main folder for pong.
* `pong_v5_script.py`: Main training script for `ALE/Pong-v5`, including CNN, frame preprocessing, stacking, and checkpointing.
* `PROBLEM-3`: Main folder for problem 3 .
* `problem_3.py`: Main script for Problem . This single script can run all PG experiments using command-line arguments.
* `batch_size`: Main folder for for problem 3(c), including checkpointing and plotting from the checkpoint.
* `report.md`: Separate Report and graphs is also attached to respective folders , as needed in the assignment .

---

## 🗂️ How to Play

To play you need to navigate to the respective folder.

* `mountain car`: navigate to mountain_car folder , need to run `play_mountain.py` in mountain_car folder and provide the absolute path to the MODEL_PATH var in the script, of the file `mountaincar_dqn.pth` .
* `Pong`: navigate to pong folder , need to run `play_pong.py` in pong folder and provide the absolute path to the MODEL_PATH var in the script, of the file `pong_dqn.pth` .
* `Cart Pole`: need to run `play_model.py` in Problem-3 Folder by navigating to that folder , and then run using the following command `python play_model.py  --model_path "/absolute/path/to/Assignment-3/PROBLEM-3/model_1_cartpole/pg_policy_net_CartPole-v1.pth" --env_name "CartPole-v1" ` .
* `Lunar radar`: need to run `play_model.py` in Problem-3 Folder by navigating to that folder , and then run using the following command `python play_model.py  --model_path "absolute/path/to/Assignment-3/PROBLEM-3/model_1_lunar_radar_5000/pg_policy_net_LunarLander-v3.pth" --env_name "LunarLander-v3" `

---

## Problem 1: Deep Q-Network (DQN)

### Part 1a: MountainCar-v0 Implementation

This script uses inbuilt random agent of the env, ,to play `ALE/Pong-v5`.

### Part 1b: MountainCar-v0 Implementation

This script uses a DQN to solve `MountainCar-v0`.

* **Network:** `QNetwork` , neural network.
* **Key Features:**
  * Uses a `ReplayBuffer` to store experiences.
  * Uses a `TargetNetwork` that updates every 10 episodes (`TARGET_UPDATE_FREQUENCY`).
  * Saves checkpoints (`mountain_dqn_checkpoint.pth`) every 500 episodes.
  * Loads from checkpoint on startup if one is found.
  * Saves the final model as `mountaincar_dqn.pth`.

**How to Run:**

```bash
# This will start training.
# If a checkpoint exists, it will resume from it.
# If you changed the network, DELETE the old checkpoint first.
python mountain_car_script.py
```

**Results:**
The network is converging at the reward b/w [(-96) and (-100)] , as expected . As per the graph `mountain_car/mountain_car_episode_reward_graph.png`.

### Part 1a: Pong-v5 Implementation

This script uses inbuilt random agent of the env ,to play `ALE/Pong-v5`.

### Part 1b: Pong-v5 Implementation

This script uses a much more complex DQN to solve `ALE/Pong-v5`.

* **Network:** `PongQNetwork` (a CNN)
* **Preprocessing:**
  * Uses `cv2` to convert `[210, 160, 3]` frames to `[84, 84]` grayscale.
  * Crops the score bar.
* **Frame Stacking:**
  * Uses a `collections.deque(maxlen=4)` to stack the 4 most recent frames.
  * The `[4, 84, 84]` stack is the "state" fed to the CNN.
* **Checkpointing:**
  * Saves checkpoints (`pong_dqn_checkpoint.pth`) periodically.
  * **Crucially, the `ReplayBuffer` is NOT saved** to prevent "Out of Memory" errors (which can be \>20GB).
  * The script uses `MIN_BUFFER_SIZE_FOR_TRAINING` to "warm up" the buffer before learning begins.

**How to Run:**

```bash
# This will start/resume training for Pong.
# This will take a VERY long time.
python pong_v5_script.py
```

**Results:**
The network is converging at the reward b/w [(+10) and (+19)] . As per the graph , `pong/pong_episode_rewards.png` .

### Part 1c: Hyper-parameter Tuning (MountainCar)

This script runs the `MountainCar-v0` training 4 times to compare different `learning_rate` settings and plots the results on a single graph.

* **Script:** `mountain_car_hyperparameter.py`
* **Settings Tested:** `[1e-5, 1e-4, 5e-4, 1e-3]`
* **Output:** Generates and saves two plots:
  1. `problem_1c_reward_comparison.png`
  2. `problem_1c_loss_comparison.png`

**How to Run:**

```bash
python mountain_car_script_hyperparameter_tuning.py
```

---

## Problem 3: Policy Gradient (Actor-Critic)

### Part 3a: LunarLander and CartPole Implementation

This script uses inbuilt random agent of the env ,to play the respective game.

This implementation uses an **Actor-Critic** model to solve Problem 3.

* **Actor (`PolicyNetwork`):** Learns the policy (which action to take).
* **Critic (`ValueNetwork`):** Learns the 'state dependent expected return' (as mentioned in the slide) , state-value $V(s)$ and acts as the "suitable baseline" to calculate the **Advantage**.
* **Advantage ($A_t$):** $A_t = G_t - V(s_t)$, where $G_t$ is the Reward-to-Go.

### Part 3b: R2G & Advantage Normalization

You can use the `problem-3.py` script and command-line flags to test all four combinations.

**How to Run (Example on `CartPole-v1`):**

```bash
# 1. Baseline (Neither feature)
python problem-3.py --env-name "CartPole-v1" --no-reward-to-go --no-advantage-norm

# 2. With R2G only
python problem-3.py --env-name "CartPole-v1" --use-reward-to-go --no-advantage-norm

# 3. With AdvNorm only
python problem-3.py --env-name "CartPole-v1" --no-reward-to-go --use-advantage-norm

# 4. With Both (Recommended)
python problem-3.py --env-name "CartPole-v1" --use-reward-to-go --use-advantage-norm
```

```bash

# For LunarLander
python problem-3.py --env-name "LunarLander-v3" --num-iterations 5000
```

similar flags can be used for lunar radar also .

**Results:**
The network is converging at the reward +500 for CartPole .
For LunarLander it is convering at the reward b/w [+150 and +180] . As per the graph `PROBLEM-3/model_1_lunar_radar_5000/pg_curve_LunarLander-v3_r2gTrue_normTrue.png`.

### Part 3c: Batch Size Experiment

You can find the `batch.sh` and `batch_lunar_radar.sh` where all the commands are given.

**Example Run:**

```bash
# For CartPole
python problem-3-batch.py --env-name "CartPole-v1" --batch-size 10
```

This will:

1. Run the experiment for `batch_size=10`.
2. Save a comparison plot (`pg_curve_CartPole-v1_batch_size_comparison.png`).
3. Save the `.pth` file .

---

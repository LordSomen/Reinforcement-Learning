### Report: Impact of Batch Size on LunarLander-v3

#### 1. Objective
To study the impact of `batch_size` on the learning performance of the Policy Gradient agent on the `LunarLander-v3` environment, per "the Assignment" Problem 3(c).

#### 2. Methodology
All experiments were run with **Reward-to-Go (R2G) = True** and **Advantage Normalization (AdvNorm) = True**. The `batch_size` (number of episodes collected before an update) was varied between 1, 5, and 10.

---

### 3. Results and Analysis

* **Batch Size = 1**
    * **Performance:** This model achieved the **highest peak reward**, successfully learning to land the craft with a smoothed reward of **over +100**.
    * **Analysis:** The learning curve is **extremely volatile** (noisy), with large dips and peaks. This shows that while the "online" updates (learning after every episode) are effective, they are also high-variance, causing unstable performance episode-to-episode.

* **Batch Size = 5**
    * **Performance:** This model represents a good compromise. It learned steadily and achieved a **positive score** (peaking around +10).
    * **Analysis:** The learning curve is **much smoother** than the batch size 1 run. Averaging gradients over 5 episodes reduced the variance (noise), leading to more stable, consistent improvement.

* **Batch Size = 10**
    * **Performance:** This model performed the **worst**.
    * **Analysis:** The learning curve is the smoothest, but it is also the slowest. The agent's performance plateaued at a negative reward (around -75), failing to learn a successful landing policy.

---

### 4. Conclusion

For the `LunarLander-v3` environment, **a small batch size (1 or 5) is far superior** to a larger one (10).

This experiment highlights the core trade-off of Policy Gradients:
* **Larger batches (e.g., 10)** produce very stable, low-variance updates, but the data becomes "stale" (high-bias), which significantly slows down or even prevents learning.
* **A tiny batch (e.g., 1)** produces high-variance, noisy updates, but the immediate feedback (low-bias) allows the agent to learn the complex task and achieve the best final performance, even if the path is unstable.
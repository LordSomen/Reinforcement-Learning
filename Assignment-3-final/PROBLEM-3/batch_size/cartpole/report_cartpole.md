### Report: Impact of Batch Size on Policy Gradient Performance

#### 1. Objective

As per Problem 3(c) of "the Assignment," this report studies the impact of **batch size** on the performance of the Policy Gradient (Actor-Critic) agent.

#### 2. Methodology

The experiment was conducted on the `CartPole-v1` environment using an Actor-Critic agent. For all runs, the following settings were held constant:

* **Reward-to-Go (R2G):** True
* **Advantage Normalization (AdvNorm):** True

The single hyper-parameter varied was the  **batch size** , which is the number of complete episodes the agent collects before performing one optimization step. The following batch sizes were tested: 1, 5, and 10.

---

### 3. Results and Analysis

Based on the three provided plots, the impact of batch size is immediate, nontrivial, and demonstrates a clear trend:

* **Batch Size = 1 (File: `...bs1...png`)**
  * **Performance:** This was the best-performing model by a significant margin.
  * **Analysis:** The learning curve is fast, stable, and rises consistently. It peaked at a smoothed reward of **over 400** by around episode 210, effectively "solving" the environment (which has a maximum reward of 500). This "online" update (learning after every single episode) provided the most effective feedback.
* **Batch Size = 5 (File: `...bs5...png`)**
  * **Performance:** This model showed a dramatic decrease in performance compared to a batch size of 1.
  * **Analysis:** The agent learned at a much slower rate, only reaching a smoothed reward of approximately **185** by episode 250. While it was still learning, its learning efficiency was severely reduced.
* **Batch Size = 10 (File: `...bs10...png`)**
  * **Performance:** This model performed the worst.
  * **Analysis:** The learning curve is very noisy and rises extremely slowly. By episode 250, it only achieved a smoothed reward of approximately  **60** , failing to learn a competent policy within the 250-episode timeframe.

---

### 4. Conclusion

For this implementation on `CartPole-v1`, **a smaller batch size is dramatically better.**

The results show a strong  **inverse correlation between batch size and performance** : as the batch size increases, the agent's ability to learn effectively decreases.

Conceptual Reason:

This demonstrates a classic Policy Gradient trade-off.

1. **With Batch Size 1 (Online):** The agent gets a noisy, high-variance gradient, but it is 100% "on-policy" (the update is for the *exact* policy that just collected the data). This high-frequency, immediate feedback was the most effective for this environment.
2. **With Batch Size 5 or 10 (Batch):** The agent averages the gradients from multiple episodes. This reduces the variance (noise) of the update but introduces **bias** (or "staleness"). The policy is being updated using data from episodes collected by *older* versions of the policy.

In this experiment, the negative impact of this "stale" data (bias) far outweighed any benefits from a lower-variance gradient.

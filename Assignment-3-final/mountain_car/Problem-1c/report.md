### Report on Hyper-parameter Tuning (Problem 1c)

**Objective:**
To analyze the impact of the **Learning Rate** hyper-parameter on the performance of the DQN agent in the `MountainCar-v0` environment, as required by "the Assignment" .

**Analysis of Learning Curves:**
Based on the provided reward and loss graphs, the learning rate had a "nontrivial difference on performance".

* **`Learning Rate = 0.001` (Red):**

  * **Result:** This was the **only learning rate that learned effectively** within the 5000 episodes, showing its first successes (rewards above -200) around episode 500.
  * **Analysis:** The learning was **highly unstable**, as shown by the extremely spiky and erratic reward and loss graphs. This suggests the agent was actively learning but constantly "overshooting" the optimal policy.
* **`Learning Rate = 0.0001` (Yellow):**

  * **Result:** This agent learned, but **very slowly**.
  * **Analysis:** It took over 3,000 episodes to achieve its first successful runs. The periodic spikes in both loss and reward show it was also unstable, though less so than the red line.
* **`Learning Rate = 1e-05` (Blue):**

  * **Result:** **Failed to learn.**
  * **Analysis:** The reward remained at -200. The loss graph shows the loss was decreasing, but at an extremely slow pace, indicating the learning rate was too small to be effective in the given number of episodes.
* **`Learning Rate = 0.0005` (Purple):**

  * **Result:** **Failed to learn.**
  * **Analysis:** The reward stayed at -200, and the loss dropped to zero almost immediately. This suggests the network converged to a bad, stable solution (likely just predicting -200) and stopped learning.

**Conclusion:**
The learning rate is a critical hyper-parameter. A rate of **`0.001`** was the most effective at solving the environment, but it was also very unstable. Rates that were too low (`1e-05`, `0.0001`) failed to learn in a reasonable time, while a rate of `0.0005` appeared to get stuck and failed entirely.

!! For computational constraint , able to perform this only for 5000 episodes , but the result provide us the necessary comparision we needed , for learning_rate's impact in the learning .

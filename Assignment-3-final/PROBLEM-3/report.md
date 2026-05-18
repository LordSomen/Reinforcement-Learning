
## Report: Impact of Variance Reduction Techniques (Problem 3b)

### Objective

To compare the learning curves of a Policy Gradient agent "with and without advantage normalization and reward-to-go functionality" on the `CartPole-v1` environment.

---

### Analysis on CartPole-v1

By comparing the four `CartPole-v1` graphs, we can analyze the impact of each technique.

* **Impact of Reward-to-Go (R2G):**

  * **`R2G=True` (File: `...r2gTrue_normTrue.png`):** The agent successfully solved the environment, reaching the maximum score of 500.
  * **`R2G=False` (File: `...r2gFalse_normTrue.png`):** The agent **failed to learn entirely**. The reward immediately dropped and stayed at a minimum value (around 9.5).
  * **Conclusion:** Using Reward-to-Go (Formulation 2)  is **essential** for this task. Using the total trajectory reward (Formulation 1) provided a poor signal that prevented learning.
* **Impact of Advantage Normalization (AdvNorm):**

  * **`AdvNorm=True` (File: `...r2gTrue_normTrue.png`):** The agent learned very quickly and stably, reaching the max score of 500 around episode 400 and staying there.
  * **`AdvNorm=False` (File: `...r2gTrue_normFalse.png`):** The agent also solved the environment, but its learning was **less stable**. The reward curve shows more "wobble" and volatility on its way to the maximum score.
  * **Conclusion:** Advantage Normalization is **highly beneficial**. It stabilizes the learning process and helps the agent converge to the optimal policy faster and more smoothly.

---

### Summary

| R2G             | AdvNorm         | Result                                                        |
| :-------------- | :-------------- | :------------------------------------------------------------ |
| **True**  | **True**  | **Best Performance:** Fast, stable, solves.             |
| **True**  | **False** | **Good Performance:** Solves, but is unstable.          |
| **False** | **False** | **Slow Performance:** Solves, but is unstable and slow. |
| **False** | **True**  | **Complete Failure:** Fails to learn.                   |

### LunarLander-v3 Confirmation

The provided `LunarLander-v3` graph (using `R2G=True`, `AdvNorm=True`) , The agent successfully learned to fly and land, achieving a positive smoothed reward of over +150.

! DUE TO COMPUTATIONAL CONSTRAINT ABLE TO RUN THIS ONLY FOR ONE SETTING.

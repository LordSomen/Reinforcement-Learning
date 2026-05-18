Answer:
(a) For any policy 

$$
\pi
$$

 and state 

$$
s
$$

, the values relate as 
$$
\hat V^{\pi}(s)=V^{\pi}(s)-\frac{\varepsilon}{1-\gamma}
$$

. [1][2]
(b) Yes, 
$$
M$$ and
$$

\hat M
$$
have the same optimal policy because subtracting the same constant from all one-step rewards translates every
$$

Q(s,a)
$$
by the same constant and leaves all
$$

\arg\max_a Q(s,a)$$ unchanged. [1]

### Setup from slides

- The policy value is 

  $$
  V^{\pi}(s)=\mathbb{E}\!\left[\sum_{k=0}^{\infty}\gamma^{k} r_{t+k+1}\,\big|\,S_t=s,\pi\right]
  $$

   with 
  $$
  0<\gamma<1
  $$

   ensuring convergence of the discounted series. [1][2]
- The action-value and greedy policy connection used to characterize optimality is 

  $$
  Q^{\pi}(s,a)=\mathbb{E}[r_{t+1}+\gamma V^{\pi}(S_{t+1})|s,a]
  $$

   and 
  $$
  \pi_{\text{greedy}}(s)\in\arg\max_a Q(s,a)
  $$

  . [1]

### Derivation for part (a)

- The modified rewards satisfy 

  $$
  \hat R(s,a,s')=R(s,a,s')-\varepsilon
  $$

  , where 
  $$
  \varepsilon
  $$

   is constant across all 
  $$
  (s,a,s')
  $$

  . [2]
- The modified return is 

  $$
  \hat G_t=\sum_{k=0}^{\infty}\gamma^{k}(r_{t+k+1}-\varepsilon)=\sum_{k=0}^{\infty}\gamma^{k}r_{t+k+1}-\varepsilon\sum_{k=0}^{\infty}\gamma^{k}
  $$

  . [1]
- Using 

  $$
  \sum_{k=0}^{\infty}\gamma^{k}=\frac{1}{1-\gamma}
  $$

   and linearity of expectation in the value definition gives 
  $$
  \hat V^{\pi}(s)=V^{\pi}(s)-\frac{\varepsilon}{1-\gamma}
  $$

   for all $$s$$. [1]

### Bellman check (equivalent argument)

- Bellman evaluation under a fixed policy is 

  $$
  V^{\pi}=R_{\pi}+\gamma P_{\pi}V^{\pi}
  $$

   and 
  $$
  \hat V^{\pi}=\hat R_{\pi}+\gamma P_{\pi}\hat V^{\pi}
  $$

   with 
  $$
  \hat R_{\pi}=R_{\pi}-\varepsilon\mathbf{1}
  $$

  . [1]
- Subtracting, 

  $$
  \Delta:=\hat V^{\pi}-V^{\pi}
  $$

   satisfies 
  $$
  \Delta=-\varepsilon\mathbf{1}+\gamma P_{\pi}\Delta
  $$

  , whose unique solution (since 
  $$
  0<\gamma<1
  $$

  ) is 
  $$
  \Delta=-\frac{\varepsilon}{1-\gamma}\mathbf{1}
  $$

  , matching the return-based derivation. [1]

### Part (b): Optimal policy invariance

- The same shift applies to action values: 

  $$
  \hat Q^{\pi}(s,a)=Q^{\pi}(s,a)-\frac{\varepsilon}{1-\gamma}
  $$

  , because 
  $$
  \hat Q^{\pi}(s,a)=\mathbb{E}[\hat r_{t+1}+\gamma \hat V^{\pi}(S_{t+1})|s,a]
  $$

  . [1]
- Since 

  $$
  \arg\max_a \hat Q(s,a)=\arg\max_a Q(s,a)
  $$

  , the greedy policy and thus the set of optimal policies are unchanged, consistent with the greedy optimality and policy-improvement results in the slides. [1]

Sources
[1] lecture4-2.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53603889/da0302cf-6403-45d9-9c45-682d9e176413/lecture4-2.pdf
[2] lecture-3-MDP-1.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53603889/6f528462-c708-412d-84d4-edd1ab1f993a/lecture-3-MDP-1.pdf
[3] Screenshot 2025-09-26 at 10.12.59 AM.png https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/53603889/a429fcf5-8f0c-47ef-867c-d163e517ac27/Screenshot-2025-09-26-at-10.12.59-AM.png?AWSAccessKeyId=ASIA2F3EMEYESTVV6VZK&Signature=RU9%2FvUBX9%2FRJizmhqiHkRTxsLpA%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEP3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDgXE%2Be5BAZGJJ247FwbvIMpDatWFzhmDn7q0NtMDklFwIhAPD1r9jqQDa%2FzyV2uQ4v5DE5rqoTDSypiQgbpkZ8XDLXKvoECIX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1Igx%2Fbdtr2A6FmvNB9QQqzgTdbXvwyZ%2FI56c6QCsiIbkHGYht2S6lNyYShWM4lHGqelA6MDcMxk3qCaNlCrezpMEfHaMYRP0xI9ehhvCQHIGkCgyy78srCXkw7mkLlvqDmq1KxViB%2BcIfPT%2FI6yHDW2XPQSi1WxuBpK6EjGYcWvbjC3%2Bfto6WP%2Bb%2FRLV3RVYtYoilYgHFPzVJ7QLR8fxJCYyi%2FRj2pVo98Hqav1sh2STSDl9Ttx80i2%2B4abfxg44fJI852gCfqsFv%2FzYBSfMsTZsqMr3FOfCAYSfaiKaiWmjoOa6pJEU0%2F3vqbL4c%2BDJhAbQKWcz%2FRmpbf9kmMR4bUvU68XqbyXruprt1%2BkHP%2BKdbyiuu%2Fj3u2%2F1SIoMFSQ23UVPHhZg%2FPBbP5crTnI0PbtNeCmUAMY2%2Fj2wqiOErLyHntTIdBDBcmz7wyg9bHeom1pwiiW8bV8FJsJGf5ZgUVgcpIAfk9l07%2FqaxsqJms38IiO3OmAfbbegSIdjz6VOKFzMnoRJW9oETxHCT4%2FkgxcQXpiOLFTxKVSOrQNxHtq7LWewhbwHqLiNqwZmz2mXRJNxoni1PZK7sWUW29R6qnrMmcHsrch3X7xP8HhxUyS0ffRAhWnONeoBLbf2DKpLvfPRLQucHSmRumzNUrCwwcKSwgJ0M5sUr7k8zWDnC6IYnz8lo36Lsf3Pr%2BhRXJeYNJoR5W7TjEQ%2BCiXp%2BsyS33dFNXNPSBDkVAKbiOjIKfS%2Fdk2ERAQ%2Fw%2BE%2Frfw2LeWpLT524YGEDGQBYu7kJlpdXN4OQ1D5kP%2FSFL7Pvnlxq7jCLptjGBjqZASbszN5o9TXR3Dhe9bj8bDeHCL3SbwthIWH3n%2FlJWuWg0RKzaSqjW%2BEtVzd4lmE4r4yKTQvpaBDoFEdyd9LzuuJ2ZzDbfqZysoTVCZxaa3bnosgFQTzfKgynt9F%2BPpRWqj6yLcmenkWq06DBulf4AcCrJObAMI31Ubideoy%2BZhjvoW6y3vDXZPi%2FR2QKCPTqt9b7B%2Fcr2lOqkA%3D%3D&Expires=1758862446

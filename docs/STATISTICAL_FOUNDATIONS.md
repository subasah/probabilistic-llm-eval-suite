# Statistical Foundations for Probabilistic LLM Evaluation

Evaluating Large Language Models requires shifting from **deterministic unit testing** to **probabilistic statistical hypothesis testing**. This document details the mathematical underpinnings implemented in `ProbEval`.

---

## 1. Why Classical Unit Tests Fail for LLMs

In classical software engineering:
$$f(x) = y \implies \text{assert } f(x) == y$$

In Large Language Models, generation is governed by token probability distributions over vocabulary $V$:
$$P(w_t \mid w_{<t}, \text{prompt}) = \text{Softmax}\left(\frac{z_t}{T}\right)$$

Even when temperature $T = 0$, floating-point non-associativity across distributed GPU tensor cores, dynamic batching, and mixture-of-experts (MoE) token routing introduce **token jitter**.

### The Flawed "Single-Sample" Trap
Consider an evaluation suite of 50 test cases. An engineer updates a system prompt. A single-shot evaluation shows:
- Run 1 (Prompt A): 45 / 50 passed (90.0%)
- Run 2 (Prompt B): 47 / 50 passed (94.0%)

**Did Prompt B genuinely improve by 4%?**
Almost certainly **no**. Under binomial sampling with $N = 50$, the 95% margin of error is approximately:
$$\text{MOE} = 1.96 \times \sqrt{\frac{p(1-p)}{N}} = 1.96 \times \sqrt{\frac{0.9 \times 0.1}{50}} \approx 8.3\%$$

The apparent 4% improvement is completely swallowed by the $\pm 8.3\%$ noise floor. Relying on single-sample evals causes teams to ship regressions while celebrating phantom gains.

---

## 2. Non-Parametric Bootstrap Resampling

To construct confidence intervals without assuming a Gaussian (normal) distribution—which is invalid for metrics bounded in $[0, 1]$ or bimodal pass/fail distributions—`ProbEval` utilizes **empirical bootstrap resampling**.

### Algorithm
Given sample evaluations $X = [x_1, x_2, \dots, x_N]$:
1. For $b = 1$ to $B$ (e.g., $B = 1500$ resamples):
   - Sample $N$ values uniformly **with replacement** from $X$:
     $$X^{*b} = [x_{1}^{*b}, x_{2}^{*b}, \dots, x_{N}^{*b}]$$
   - Compute the bootstrap replicate statistic:
     $$\hat{\theta}^{*b} = \frac{1}{N} \sum_{i=1}^N x_{i}^{*b}$$
2. Sort the bootstrap replicates: $\hat{\theta}^{*(1)} \le \hat{\theta}^{*(2)} \le \dots \le \hat{\theta}^{*(B)}$.
3. For confidence level $1 - \alpha$ (e.g., $\alpha = 0.05$ for 95% CI):
   $$\text{CI}_{\text{lower}} = \hat{\theta}^{*\left(\left\lfloor \frac{\alpha}{2} B \right\rfloor\right)}, \quad \text{CI}_{\text{upper}} = \hat{\theta}^{*\left(\left\lceil \left(1 - \frac{\alpha}{2}\right) B \right\rceil\right)}$$

---

## 3. Wilson Score Interval for Small Sample Proportions

When sample sizes are small or the pass rate is close to 0 or 1, normal approximations fail catastrophically (predicting confidence intervals $> 1.0$ or $< 0.0$).

The **Wilson Score Interval** provides an exact, asymmetric interval:
$$w = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}$$

where:
- $\hat{p} = \frac{c}{N}$ is the observed sample pass rate
- $z$ is the $1 - \frac{\alpha}{2}$ quantile of the standard normal distribution ($z \approx 1.96$ for 95%)

---

## 4. Unbiased Pass@k Estimation

When evaluating generative tasks (e.g., code generation or multi-step reasoning), measuring $\text{pass@}1$ only captures the probability that the very first attempt succeeds. In real systems, workflows may generate $k$ attempts and execute unit tests or verifiers.

If we draw $n$ samples per prompt and $c$ succeed, a naive calculation of $\text{pass@}k$ on subsets exhibits high variance. Following **Chen et al. (2021) [HumanEval]**, `ProbEval` uses the unbiased minimum-variance estimator:

$$\text{pass@}k = \mathbb{E}\left[ 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} \right]$$

To prevent numerical overflow from large factorials, `ProbEval` computes this via the stable sequential product:
$$\text{pass@}k = 1 - \prod_{i=1}^k \frac{n - c - i + 1}{n - i + 1}$$

### Intuition:
- If $n - c < k$: There are fewer failures than the budget $k$, meaning at least one attempt must pass $\implies \text{pass@}k = 1.0$.
- If $c = 0$: No attempts passed $\implies \text{pass@}k = 0.0$.
- For intermediate values, it directly quantifies how multi-attempt sampling recovers capabilities masked by single-token variance.

---

## 5. McNemar's Paired Test for Categorical Regressions

When testing whether Prompt Candidate $B$ is better or worse than Baseline $A$ on the **exact same set of benchmark prompts**, an independent two-sample test (like chi-squared) is inappropriate because **the test cases are paired**.

### Contingency Table of Paired Outcomes:
$$\begin{array}{c|cc}
& \text{Candidate Pass} & \text{Candidate Fail} \\
\hline
\text{Baseline Pass} & a \text{ (Concordant Pass)} & b \text{ (Regressions!)} \\
\text{Baseline Fail} & c \text{ (Improvements!)} & d \text{ (Concordant Fail)}
\end{array}$$

Notice that $a$ and $d$ represent queries where both models agreed. The statistical question is entirely determined by **discordant pairs**:
- $b$: Prompts that passed in Baseline, but **failed** in Candidate (regressions).
- $c$: Prompts that failed in Baseline, but **passed** in Candidate (improvements).

### Hypothesis:
- $H_0: P(b) = P(c)$ (The change causes no net difference; regressions and improvements are equal noise).
- $H_1: P(b) \ne P(c)$ (The change causes a statistically significant shift).

### Test Statistic:
For large discordant samples ($b + c \ge 25$), Edwards' continuity-corrected chi-squared statistic with 1 degree of freedom is:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

For small discordant samples ($b + c < 25$), `ProbEval` switches automatically to the **Exact Two-Sided Binomial Test**:
$$p = 2 \times \sum_{i=0}^{\min(b, c)} \binom{b + c}{i} 0.5^{b + c}$$

If $p < 0.05$ and $b > c$, a **statistically significant regression** is proven, and the CI gate terminates the deployment.

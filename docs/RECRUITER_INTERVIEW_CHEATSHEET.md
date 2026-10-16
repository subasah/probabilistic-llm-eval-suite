# Recruiter & Technical Interview Cheat Sheet: LLM Evaluation & Regression Testing

This guide prepares you to speak authoritatively about this project in AI Engineering, LLMOps, and Machine Learning System Design interviews.

---

## 1. The 30-Second Elevator Pitch

> *"Most teams evaluate LLMs with single-shot prompts and string assertions, which produces flaky tests and allows silent regressions to reach production. In `ProbEval`, I built an automated evaluation harness and continuous regression gate that treats LLMs as probabilistic systems. It uses repeated stochastic sampling, bootstrap 95% confidence intervals, unbiased Pass@k estimation, and McNemar's paired hypothesis testing in CI/CD to distinguish random sampling noise from true model degradation."*

---

## 2. Core Concepts: Top 7 Interview Questions & Model Answers

### Q1: "Why can't you just use traditional unit tests (e.g. `assert response == expected`) for LLM applications?"
**Answer:**
"Generative LLMs are probabilistic token sampling engines. Even with temperature set to zero, GPU floating-point non-determinism, dynamic batching, and mixture-of-experts routing cause minor token jitter. Furthermore, natural language allows infinite valid semantic permutations for the same intent. 

If you use rigid string assertions, you either get flaky builds that block developers or engineers weaken the assertions to the point where they miss real regressions. Evaluation must be multi-tiered: deterministic schema validation for structure, embedding cosine similarity and token F1 for semantics, and statistical hypothesis testing over multiple samples to establish confidence."

---

### Q2: "What is a 'silent regression' in an LLM pipeline and how did you prevent it?"
**Answer:**
"A silent regression happens when an engineer modifies a system prompt or switches model checkpoints to fix a specific customer complaint, but that change inadvertently degrades performance on 5% of edge cases elsewhere in the system. Because no hard error is thrown, the bug slips into production.

In `ProbEval`, I prevented this with a Continuous Regression Gatekeeper in CI/CD. Every PR is evaluated against a golden baseline using McNemar's paired test. We look specifically at the discordant pairs—cases that previously passed but now fail. If the net regression exceeds our effect-size threshold ($\epsilon = 3\%$) with statistical significance ($p < 0.05$), the build is blocked automatically."

---

### Q3: "Why did you choose McNemar's test instead of a standard Chi-Squared or Two-Sample t-test?"
**Answer:**
"A standard two-sample test assumes independent samples. But in benchmark regression testing, the samples are **paired**: we are testing the baseline model and candidate model on the **exact same set of prompt queries**. 

Most of the queries will either pass on both models or fail on both models (concordant pairs). An independent chi-square test is diluted by all the concordant items and lacks statistical power. McNemar's test focuses exclusively on the discordant pairs (where one model passed and the other failed), using Edwards' continuity correction or an exact binomial distribution to determine if the regressions significantly outnumber the improvements."

---

### Q4: "What is Position Bias in LLM-as-a-judge and how did you eliminate it?"
**Answer:**
"Position bias is the systematic tendency for an LLM judge to favor the first option presented (Option A) by 10% to 20%, regardless of output quality. 

To solve this, I designed a **Bidirectional Swapped Pairwise Judge**. For every pair of outputs, we evaluate twice: once with Candidate as Option A and Baseline as Option B, and once with Baseline as Option A and Candidate as Option B. 
A candidate only receives a win if it wins in both orientations. If the judge picks Option A in both orientations, our system detects this as an artifact of position bias, discounts the verdict, and marks it as a calibrated tie."

---

### Q5: "What is Pass@k and why is the naive calculation biased?"
**Answer:**
"Pass@k measures the probability that at least one of $k$ generated samples passes validation. If you generate $n$ samples (e.g. 10) and take random subsets of size $k$ (e.g. 3) to see if at least one passes, the estimate has high variance and is biased. 

Following the formulation in the HumanEval paper by Chen et al. (2021), `ProbEval` uses the unbiased minimum-variance estimator:
$$\text{pass@}k = 1 - \prod_{i=1}^k \frac{n - c - i + 1}{n - i + 1}$$
where $n$ is total samples and $c$ is the number of passing samples. This allows us to accurately estimate the multi-sample yield of a model without running combinatorial evaluations."

---

### Q6: "Why use Bootstrap Resampling instead of calculating Standard Deviation with Gaussian assumptions?"
**Answer:**
"Standard parametric confidence intervals assume that sample errors follow a normal Gaussian distribution ($\mu \pm 1.96 \frac{\sigma}{\sqrt{N}}$). However, evaluation metrics like pass rates, exact match, and schema validity are strictly bounded in $[0, 1]$ and often follow bimodal or heavily skewed distributions. A Gaussian interval will frequently produce invalid confidence intervals (e.g. upper bounds $> 100\%$). 

Non-parametric bootstrap resampling draws 1,500 replicates with replacement from the empirical data distribution itself, providing valid percentile-based 95% confidence intervals without making distribution assumptions."

---

### Q7: "How do you control the cost and latency of running LLM evaluations in CI/CD?"
**Answer:**
"We implement a **Hierarchical Evaluation Funnel**:
1. **Tier 1 (Zero Cost, <1ms)**: Deterministic validation runs first. If a candidate output fails JSON schema validation or regex guardrails, it fails immediately without invoking expensive models.
2. **Tier 2 (Zero Cost, <5ms)**: Token F1 and n-gram semantic similarity checks.
3. **Tier 3 (Model Calls)**: Calibrated LLM judges are reserved only for complex reasoning and subjective criteria.
4. **Sampling Budgets**: In fast PR check workflows, we evaluate with $n=3$ to $5$ samples per prompt on a curated core benchmark suite (~30 high-signal prompts). Nightly scheduled runs evaluate comprehensive suites with $n=15$ samples across full datasets."

---

## 3. System Design Topic: "Design an Enterprise Continuous Evaluation Platform"

When asked to design an LLM evaluation system at scale:

1. **Ingestion & Curation**:
   - Curated Golden Datasets (human-verified ground truths) + Production Traffic Shadowing (sampling real user queries with PII scrubbing).
2. **Execution Worker Pool**:
   - Asynchronous worker queue (Celery / Temporal / Asyncio) with distributed rate limiters to avoid hitting model provider TPM/RPM limits.
3. **Evaluation Matrix**:
   - Deterministic checks $\to$ Semantic heuristics $\to$ LLM-as-a-judge $\to$ Human-in-the-loop review queue for low-confidence discordant judgments.
4. **Statistical Gating**:
   - Baseline regression store, McNemar test, bootstrap confidence intervals.
5. **Observability & Dashboards**:
   - Interactive HTML reports, Slack/PagerDuty alerts for critical regressions, GitHub PR status checks.

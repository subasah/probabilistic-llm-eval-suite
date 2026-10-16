# System Architecture & Technical Design

`ProbEval` is a production-grade framework designed to solve the central challenge of LLM quality engineering: **evaluating and regression-testing probabilistic, non-deterministic model outputs with statistical rigor.**

---

## 1. High-Level Architectural Flow

The architecture operates across five distinct phases:
1. **Benchmark Specification**: Ingestion of test suites (structured JSON, RAG contexts, tool schemas, reasoning prompts).
2. **Stochastic Sampling & Execution Engine**: Asynchronous concurrent generation of repeated samples ($n$ trials per prompt) across temperature curves.
3. **Multi-Tiered Evaluation Matrix**: Fast-path deterministic checks, semantic embeddings, and calibrated LLM judges.
4. **Statistical Inference Engine**: Bootstrap resampling, Pass@k estimation, and McNemar hypothesis testing.
5. **Continuous Quality Gate & Reporting**: CI/CD regression gating, terminal summaries, and interactive HTML dashboards.

```mermaid
flowchart TD
    subgraph Benchmark ["1. Benchmark Ingestion"]
        TC[Test Cases & Schemas]
        DS1[(Structured Extraction)]
        DS2[(RAG Faithfulness)]
        DS3[(Tool Calling Specs)]
        DS4[(Code & Reasoning)]
        DS1 & DS2 & DS3 & DS4 --> TC
    end

    subgraph Execution ["2. Stochastic Execution Engine"]
        TC --> Engine[Async Parallel Engine]
        Engine -->|n samples per prompt| Gen[Multi-Sample Generation]
        Model[LLM Provider / Simulator] <--> Gen
    end

    subgraph Evaluators ["3. Multi-Tiered Evaluator Matrix"]
        Gen --> EvalMatrix{Evaluator Pipeline}
        EvalMatrix --> Det[Deterministic Tier<br/>• JSON Schema<br/>• Regex Match<br/>• Tool Call Arguments<br/>• Latency SLA]
        EvalMatrix --> Sem[Semantic Tier<br/>• Token F1 SQuAD<br/>• N-gram Cosine Similarity<br/>• Normalized Exact Match]
        EvalMatrix --> Judge[Calibrated Judge Tier<br/>• Rubric Evaluation<br/>• Bidirectional Swapped Pairwise]
        EvalMatrix --> RAG[RAG Triad Tier<br/>• Context Faithfulness<br/>• Answer Relevance]
    end

    subgraph Statistics ["4. Statistical Inference Engine"]
        Det & Sem & Judge & RAG --> StatEngine[Statistical Rigor Core]
        StatEngine --> Boot[Bootstrap 95% Confidence Intervals]
        StatEngine --> PassK[Unbiased Pass@k Estimator]
        StatEngine --> McNemar[McNemar Paired Test on Discordant Cases]
        StatEngine --> Welch[Welch t-test on Continuous Metrics]
    end

    subgraph Gatekeeper ["5. Continuous Regression Gating & Delivery"]
        StatEngine --> Gate[Continuous Regression Gatekeeper]
        Base[(Golden Baseline)] -.-> Gate
        Gate -->|Δ > ε and p < 0.05| Fail[🚨 Critical Regression: Block Build]
        Gate -->|Δ within noise floor| Pass[✅ Quality Gate Passed]
        Gate --> Report[Console & Interactive HTML Dashboard]
    end
```

---

## 2. Component Breakdown

### 2.1 Model Provider Abstraction (`src/probeval/models/`)
- **`BaseModelProvider`**: Protocol specifying `generate_async()` and `generate()`, returning unified `ModelResponse` with normalized text, token counts, latency, and tool calls.
- **`SimulatedLLMProvider`**: Zero-cost, reproducible LLM simulator that exhibits authentic probabilistic non-determinism, controllable temperature noise, and parameterized regression injection.
- **`OpenAIProvider` & `ModelFactory`**: Direct support for OpenAI, Anthropic, Gemini, or any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter).

### 2.2 Multi-Tiered Evaluators (`src/probeval/evaluators/`)
Evaluating generative outputs with a single metric produces blind spots. `ProbEval` organizes evaluators into tiered costs and guarantees:

| Tier | Evaluator | Typical Latency | Cost | Failure Mode Caught |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Deterministic** | `JsonSchemaEvaluator` | < 1 ms | $0 | Syntax errors, missing required keys, schema violations |
| **Tier 1: Deterministic** | `ToolCallEvaluator` | < 1 ms | $0 | Hallucinated function names, invalid parameter JSON |
| **Tier 1: Deterministic** | `RegexEvaluator` | < 1 ms | $0 | Banned toxic phrases, missing mandatory terms |
| **Tier 1: Deterministic** | `LatencyBudgetEvaluator`| < 0.1 ms | $0 | SLA latency spikes and inference degradation |
| **Tier 2: Semantic** | `TokenF1Evaluator` | < 2 ms | $0 | Paraphrasing quality, token overlap precision/recall |
| **Tier 2: Semantic** | `CosineSimilarityEvaluator`| < 5 ms | $0 | Semantic drift, vector proximity |
| **Tier 3: RAG Triad** | `FaithfulnessEvaluator` | < 5 ms | $0 | Hallucinated claims unsupported by retrieved context |
| **Tier 3: RAG Triad** | `AnswerRelevanceEvaluator`| < 5 ms | $0 | Dodging questions, conversational off-topic drift |
| **Tier 4: Model Judge** | `CalibratedRubricJudge` | ~500 ms | API Cost | Nuanced reasoning, adherence to complex guidelines |
| **Tier 4: Model Judge** | `PairwiseJudge` (Swapped)| ~1000 ms | API Cost | Head-to-head model ranking with position bias mitigation |

---

## 3. The Continuous Regression Lifecycle

In classical CI/CD, a test is binary (`assert result == expected`). In LLMOps, prompt changes, temperature sampling, and model weight updates introduce **stochastic output variation**.

```mermaid
sequenceDiagram
    autonumber
    actor Dev as AI Engineer
    participant Git as GitHub CI Action
    participant Runner as ProbEval Runner
    participant Model as Candidate LLM Release
    participant Base as Golden Baseline Run
    participant Gate as Regression Gatekeeper

    Dev->>Git: Push PR with Updated Prompt v2
    Git->>Runner: Trigger Continuous Regression Gate
    Runner->>Model: Execute 5 repeated samples across benchmark suite
    Model-->>Runner: Return probabilistic response distributions
    Runner->>Runner: Compute bootstrap 95% CIs & metric means
    Runner->>Base: Fetch Golden Baseline Report
    Runner->>Gate: Compare Baseline vs Candidate
    Gate->>Gate: Run McNemar's Test on discordant test cases
    Gate->>Gate: Run Welch t-test on continuous metric deltas
    alt Statistically Significant Drop (p < 0.05, Δ < -3%)
        Gate-->>Git: 🚨 Quality Gate FAILED (Exit Code 1)
        Git-->>Dev: Post PR Failure Breakdown & Discordant Regressions
    else Shift within Noise Floor (p >= 0.05)
        Gate-->>Git: ✅ Quality Gate PASSED (Noise Accepted)
        Git-->>Dev: Post PR Approval & HTML Dashboard Link
    end
```

---

## 4. Key Design Principles

1. **Deterministic Reproducibility**: The evaluation harness supports fixed random seeds for both simulator sampling and bootstrap resamples, enabling deterministic debugging of stochastic systems.
2. **Zero-Lockin Architecture**: Core statistical engines and deterministic evaluators have zero external cloud dependencies.
3. **Fail-Fast Gating**: Critical evaluators (e.g., schema validation) can be designated as blocking, immediately flagging regressions before downstream steps run.
4. **Observable Artifacts**: Evaluation reports are compiled into standalone HTML files with inline data, ensuring they can be viewed offline, attached to GitHub Actions, or embedded in internal engineering portals.

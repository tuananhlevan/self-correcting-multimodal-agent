# Multimodal Self-Correcting Reasoning Agent

An autonomous, self-correcting vision agent that extracts data from complex charts and writes deterministic Python scripts to verify its own analytical reasoning. Built entirely with local open-weights models.

## The Problem & The Solution

**The Problem:** Standard Vision-Language Models (VLMs) hallucinate math. When asked to calculate a percentage drop on a complex bar chart, they often guess the final number rather than performing the calculation, leading to high failure rates on analytical benchmarks.

**The Solution:** This project implements a **"System 2" reasoning loop** using a Directed Cyclic Graph. 
1. The agent uses Chain-of-Thought token conditioning to extract raw variables from the image.
2. It writes a deterministic Python verification script using the extracted data.
3. It executes the script in an isolated, timeout-protected sandbox.
4. **Self-Correction:** If the code crashes or the math fails, the agent captures the `stderr` stack trace, loops backward, evaluates its previous failure, and re-reads the chart to fix its mistake.

## System Architecture



The orchestration is handled via **LangGraph**, utilizing a strict state machine to manage memory and prevent infinite loops.

* **Unified AI Engine:** `Qwen/Qwen2.5-VL-7B-Instruct` handles both visual extraction and code generation. Performed `4bit-quantized` and loaded in `torch.bfloat16` to strictly fit within a 16GB VRAM constraint (~15GB actual footprint).
* **Control Flow:** Deterministic routing based on sandbox execution `returncode`, bypassing unreliable LLM function-calling for loop management.
* **Fail-Safes:** Includes regex fallback parsers for JSON hallucination and strict process isolation (`subprocess`) for executing generated code.

## Evaluation & Benchmarks

The self-correction loop was rigorously evaluated against a 100-sample validation subset of the **HuggingFace ChartQA** dataset.

| Metric | Baseline (Zero-Shot) | Self-Correcting Loop (System 2) | Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 57% | **63%** | 🟢 **+6%** |
| **Avg. Inference Time** | 5.6s | 14.2s | - |
| **Peak VRAM** | 15.1 GB | 15.1 GB | Maintained |

*Note: The unified singleton pattern ensures peak VRAM remains stable even during cyclic prompt loops.*

## Tech Stack
* **Deep Learing:** `PyTorch`, `transformers`, `accelerate`, `bitsandbytes`
* **Agentic Framework:** `LangGraph`
* **Frontend/Observability:** `Streamlit`
* **Evaluation:** `datasets`

## Quick Start

**1. Clone & Install**
```bash
git clone https://github.com/tuananhlevan/self-correcting-multimodal-agent
cd self-correcting-multimodal-agent
```

**2. Create conda environment**
```bash
conda env create -f environment.yaml
conda activate vqa
```

**3. Run the Interactive Dashboard**
To visualize the agent's thought process, stack trace analysis, and self-correction loops in real-time:
```bash
streamlit run ui/app.py
```

**4. Run the Evaluation Suite**
```bash
python -m evals.generate_dataset    # Generate evaluating dataset
python -m evals.benchmark           # Run the evaluation
```
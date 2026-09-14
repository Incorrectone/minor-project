# An LLM based Pipeline for deterministic Text generation

This repository contains the source code and experimental framework for my Minor Project. The project investigates GPU batch-induced non-determinism in Large Language Models (LLMs) and proposes a pipeline to evaluate and mitigate these effects during structured text generation.

## Project Overview

In production environments, LLM requests are frequently batched to maximize GPU utilization. However, due to floating-point rounding errors and non-associative matrix multiplications inherent to CUDA architectures, high batch sizes can introduce probabilistic non-determinism. This means that an LLM with a temperature of 0.0 can produce different outputs for the exact same prompt, strictly depending on the concurrent batch load.

This project measures the precise impact of this hardware-level instability on structured text generation, specifically focusing on generating rigid XML formats using grammar constraints.

### Experimental Variables
- **Models Evaluated:** Qwen2.5 (3B & 7B) and Llama-3.2 (3B)
- **Batch Configurations:** 1, 4, 8, and 16 concurrent sequences
- **Generation Constraints:** GBNF (Grammar-Based Network Format) vs. Unconstrained Generation
- **Hardware Baselines:** GPU Execution vs. Deterministic CPU Execution

## Repository Structure

- `03_experiment_runner.py`: The core execution script. It utilizes a thread pool to send concurrent requests to a local `llama.cpp` server, successfully simulating production-level batching pressure.
- `04_analyze_results.py`: A data analysis script utilizing Pandas to aggregate the experimental data and calculate exact-match accuracy and stability metrics.
- `windows_manager.py`: A cross-environment orchestration script. To bypass WSL memory limitations during execution, this script runs natively on Windows and allows the WSL pipeline to automatically spawn and manage the Windows-native GPU server via HTTP requests.
- `ticket_schema.gbnf`: The grammar ruleset used to enforce strict XML output during the constrained generation tests.
- `llm_context/`: Contains the synthetic query dataset and the overarching methodology notes.

## Setup and Execution

### Prerequisites
1. Download the required Q4_K_M GGUF models and place them in the `models/` directory.
2. Install `llama.cpp` natively on Windows.

### Running the Pipeline
Due to WSL memory allocation constraints, the execution pipeline is split across both the Windows host and the WSL environment.

1. Open a native Windows PowerShell and start the management server:
```powershell
python windows_manager.py
```

2. Open your WSL terminal and execute the experiment runner:
```bash
python 03_experiment_runner.py
```
*Note: The runner will automatically communicate with the Windows manager to cycle through all necessary models and batch sizes.*

### Data Analysis
Once the execution script generates the `experiment_results.csv` file, run the analysis script to compute the final stability and accuracy metrics:
```bash
python 04_analyze_results.py
```
This will output a detailed breakdown of how batch sizing and grammar constraints impact the deterministic reliability of the models.

# Flexive Training

This repository contains the code implementation for FlexiVe, a flexible verification system for enhancing large language model reasoning at test time.

## Abstract

Large Language Model (LLM) reasoning for complex tasks inherently involves a
trade-off between solution accuracy and computational efficiency. The subsequent
step of verification, while intended to improve performance, further complicates
this landscape by introducing its own challenging trade-off: sophisticated Generative Reward Models (GenRMs) can be computationally prohibitive if naively
integrated with LLMs at test-time, while simpler, faster methods may lack reliability. To overcome these challenges, we introduce FlexiVe, a novel generative
verifier that flexibly balances computational resources between rapid, reliable "fast
thinking" and meticulous "slow thinking" using a Flexible Allocation of Verification Budget strategy. We further propose the Solve-Detect-Verify pipeline, an
efficient inference-time scaling framework that intelligently integrates FlexiVe,
proactively identifying solution completion points to trigger targeted verification
and provide focused solver feedback. Experiments show FlexiVe achieves superior
accuracy in pinpointing errors within reasoning traces on ProcessBench. Furthermore, on challenging mathematical reasoning benchmarks (AIME 2024, AIME
2025, and CNMO), our full approach outperforms baselines like self-consistency
in reasoning accuracy and inference efficiency. Our system offers a scalable and
effective solution to enhance LLM reasoning at test time.

## Overview

The repository contains two main components:

1. **FlexiVe Training**: Code for training the flexible verification model using GRPO
2. **Solve-Detect-Verify Pipeline**: Implementation of our efficient inference-time scaling framework

## Training the Verification Model

The training code uses GRPO to fine-tune large language models to detect errors in multi-step solutions. The core of the approach is a reward mechanism that incentivizes the model to:

1. Correctly identify which paragraph in a solution contains an error
2. Return -1 when no error exists
3. Provide appropriate reasoning lengths based on problem complexity

### Setup

1. Install dependencies:
```bash
pip install transformers trl accelerate deepspeed bitsandbytes datasets matplotlib scikit-learn
```

2. Prepare data:
```bash
python data_prep.py
```

3. Configure DeepSpeed acceleration:
The repository includes configuration files for DeepSpeed and Accelerate.

### Training

To train the model with default parameters:
```bash
bash run_grpo_train.sh
```

For custom training configuration:
```bash
bash run_grpo_train.sh --model MODEL_PATH --output OUTPUT_DIR
```

## Solve-Detect-Verify Pipeline

Our implementation (`solve_detect_verify.py`) offers a complete inference-time framework that:

1. Efficiently solves complex reasoning tasks using LLMs
2. Detects hesitation points and solution completion
3. Integrates flexible verification to identify errors and guide corrections
4. Balances computational resources between fast and slow thinking
```python
python test_time_scaling_seq_mega_step.py --num_processes 12 \\
                                          --apply_detection \\
                                          --adaptive_verification \\
                                          --apply_verification \\
                                          --output result_dyve_adaptive_fuzzy_k8_aime2024.json \\
                                          --verification_k 4 \\
                                          --problems_path ./aime_2024.jsonl \\
                                          --majority_voting_n 4 
```

### Key Features

- Adaptive verification scheme with configurable thresholds
- Detection of solver hesitation points for targeted verification
- Efficient computation through intelligent KV cache reuse
- Support for best-of-N sampling and majority voting

## Key Components

- `grpo_train.py`: Main training script for FlexiVe
- `solve_detect_verify.py`: Inference-time scaling framework
- `data_prep.py`: Data preparation utilities
- `templates/`: Prompt templates
- Configuration files for distributed training

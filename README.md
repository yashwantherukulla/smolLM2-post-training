# GRPO with XML Format on GSM8K

This repository implements **Group Relative Policy Optimization (GRPO)** for training language models to solve math word problems from the GSM8K dataset using a structured XML-like reasoning format.

## 📋 Project Overview

The project fine-tunes the **SmolLM2-135M-Instruct** model to generate step-by-step reasoning in a structured XML format before providing final answers. The training uses GRPO, a reinforcement learning approach that optimizes policy based on multiple reward functions.

### Key Features:
- **Structured Reasoning**: Models learn to output reasoning in `<reasoning>` and `<answer>` XML tags
- **Multi-Objective Rewards**: Combines correctness, formatting, and structural rewards
- **Parameter-Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation)
- **Comprehensive Evaluation**: Tracks multiple metrics during training

## 🏗️ Architecture

### Model & Training
- **Base Model**: `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Method**: Group Relative Policy Optimization (GRPO)
- **Fine-tuning**: LoRA with rank 16
- **Dataset**: GSM8K (OpenAI's grade school math problems)

### Reward Functions
The training optimizes for four objectives:

1. **Correctness Reward** (2.0): Matches final answer with ground truth
2. **Format Reward** (0.5): Ensures proper XML structure
3. **Integer Reward** (0.5): Encourages numerical answers
4. **XML Count Reward** (0.5): Rewards proper tag usage and penalizes duplicates

## 📁 Repository Structure

```
grpo_xml_gsm8k/
├── grpo_xml_gsm8k.ipynb          # Main training notebook
├── outputs/
│   └── default-GRPO/             # Training outputs and checkpoints
│       └── final_model/          # Final trained model
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Installation
```bash
pip install trl peft transformers datasets torch wandb
```

### Training
The main training pipeline is in `grpo_xml_gsm8k.ipynb`:

1. **Data Preparation**: Loads and formats GSM8K dataset with XML structure
2. **Model Setup**: Initializes base model with LoRA configuration
3. **GRPO Training**: Runs policy optimization with multiple reward functions
4. **Evaluation**: Monitors training progress and model performance

## 📊 Results

Training progress and results are tracked using Weights & Biases:

<img width="1819" height="620" alt="image" src="https://github.com/user-attachments/assets/41baf7ba-703c-4b6c-b180-114fb8bc0523" />

## 🎯 Expected Output Format

The model learns to generate responses in this structured format:

```xml
<reasoning>
Step-by-step mathematical reasoning goes here.
Show the calculation process clearly.
</reasoning>
<answer>
Final numerical answer only
</answer>
```

## 🔧 Customization

### Modifying Reward Weights
Adjust the balance between different objectives in `TrainingConfig`:
```python
config = TrainingConfig(
    correctness_reward=2.0,    # Emphasize correct answers
    format_reward=0.5,         # Moderate formatting importance
    xmlcount_weight=0.5        # Structural consistency
)
```

### Changing Model Architecture
Switch to different base models by modifying:
```python
config.model_name = "your-preferred-model"
```

## 📈 Performance

The model shows improved:
- **Structured reasoning** capabilities
- **Consistent formatting** in responses
- **Step-by-step problem solving** skills
- **Answer accuracy** on mathematical problems but still gets most of the questions **wrong**. This could be because of the base model only having 135M parameters. 

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional reward functions
- Adding KL pernalty to not make the model reward hack
- Different base models
- Alternative training strategies
---

*For detailed training logs and interactive visualizations, check the Weights & Biases project dashboard.*

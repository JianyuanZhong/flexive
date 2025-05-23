import os
import argparse
import torch
import json
import glob
import numpy as np
import re
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from functools import partial

from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
For maximum efficiency, run GRPO with vLLM server:

1. Launch vLLM server first:
   ```
   trl vllm-serve --model model-7b --tensor_parallel_size 1
   ```

2. Then run this script with use_vllm=True (already enabled in configuration)

This allows scaling efficiently with models exceeding 70B parameters and supports multi-node training.
"""

def extract_answer(solution_text: str):
    """Extract the answer from the model's response using regex patterns."""
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    
    # Try to find a numeric answer if no boxed answer is found
    if "index of -1" in solution_text.lower() or "index: -1" in solution_text.lower():
        return "-1"
    
    # Look for paragraph indices
    paragraph_pattern = r'paragraph[\s_]*(\d+)'
    paragraph_matches = re.findall(paragraph_pattern, solution_text.lower())
    if paragraph_matches:
        return paragraph_matches[0]
    
    # Check for direct indices
    index_pattern = r'index[\s:]*(is|of)?[\s:]*(-?\d+)'
    index_matches = re.findall(index_pattern, solution_text.lower())
    if index_matches:
        for match in index_matches:
            return match[1]
    
    return None

def load_mistake_data(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Convert None to -1 for consistency
                if item.get('mistake_index') is None:
                    item['mistake_index'] = -1
                data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON in {file_path}")
                continue
    return data

def prepare_input_mistake(template, input_d):
    """Prepare input for the mistake detection task."""
    problem = input_d['input']
    steps = input_d['steps']
    
    # Format the steps with tags for paragraph identification
    tagged_steps = ''
    for sdx, step in enumerate(steps):
        tagged_steps += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_steps = tagged_steps.strip()
    
    # Create the formatted prompt using the template
    prompt = template.format(problem=problem, tagged_response=tagged_steps)
    return prompt

def compute_reward(prediction, target):
    """
    Compute the reward for a prediction compared to the target.
    
    Returns:
    - 1.0 for exact match
    - 0.5 for partial match (e.g., correctly identifying presence of mistake but wrong index)
    - 0.0 for complete mismatch
    """
    if prediction is None:
        return 0.0
    
    try:
        pred = int(prediction)
        targ = int(target)
        
        if pred == targ:
            return 1.0
        # Partial credit for correctly identifying whether there's a mistake at all
        elif (pred == -1 and targ == -1) or (pred != -1 and targ != -1):
            return 0.5
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0

def preprocess_function(examples, tokenizer, template, max_length=2048):
    """Process examples for model training."""
    # List to store processed inputs
    # input_ids_list = []
    # attention_mask_list = []
    # labels_list = []

    prompt_list = []
    groundtruth_list = []
    
    for example in examples["data"]:
        # Prepare the prompt
        prompt = prepare_input_mistake(template, example)
        messages = [{"role": "user", "content": prompt}]
        
        # Format using chat template
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompt_text += "\nOkay, I think I have finished thinking.\n</think>\n\n"

        prompt_list.append(prompt_text)
        groundtruth_list.append(example["mistake_index"])
        
        # # Tokenize
        # encoded = tokenizer(
        #     prompt_text, 
        #     max_length=max_length, 
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # )
        
        # input_ids_list.append(encoded["input_ids"][0])
        # attention_mask_list.append(encoded["attention_mask"][0])
        # labels_list.append(encoded["input_ids"][0].clone())
    
    # Create processed features
    result = {
        "prompt": prompt_list,
        "ground_truth": groundtruth_list,
        "original_example": examples["data"]
    }
    
    return result

class SaveBestModelCallback(TrainerCallback):
    """Callback to save best model based on average reward."""
    def __init__(self):
        self.best_reward = -float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_reward = metrics.get("eval_reward", 0)
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            # Save the best model
            output_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the model from kwargs
            trainer = kwargs.get("trainer")
            if trainer:
                trainer.save_model(output_dir)
                logger.info(f"Saved best model with reward {current_reward}")

def reward_func(completions, ground_truth, **kwargs):
    """
    Compute rewards by comparing model completions to ground truth.
    
    Args:
        completions: List of model completion strings
        ground_truth: List of ground truth values
        **kwargs: Additional arguments
    
    Returns:
        torch.Tensor: Tensor of rewards
    """
    rewards = []
    
    for completion, target in zip(completions, ground_truth):
        # Extract model's prediction from the completion
        prediction = extract_answer(completion)
        
        # Convert target if it's a tensor
        if isinstance(target, torch.Tensor):
            target = target.item()
        
        # Compute reward
        reward = compute_reward(prediction, target)
        rewards.append(torch.tensor(reward))
    
    # Create a tensor of rewards with correct shape for GRPOTrainer
    rewards_tensor = torch.stack(rewards)
    
    # Log some information about rewards
    if len(rewards) > 0:
        logger.info(f"Base reward function - Mean reward: {rewards_tensor.mean().item():.4f}")
    
    return rewards_tensor

def length_adjustment_reward_func(completions, ground_truth, **kwargs):
    """
    Calculates a reward adjustment based on response length.
    Promotes:
    1. Fast thinking (short responses) for examples with no mistakes (target == -1)
    2. Slow thinking (detailed responses) for examples with mistakes (target != -1)

    Args:
        completions: List of model completion strings
        ground_truth: List of ground truth values (mistake indices)
        **kwargs: Must include 'tokenizer'

    Returns:
        torch.Tensor: Length-based reward adjustments for each completion
    """
    adjustments = []
    tokenizer = kwargs.get('tokenizer')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("model-14b")

    # Define target lengths
    target_fast_len = 50  # Target token count for "no mistake" response
    target_slow_len = 512  # Target token count for "mistake found" response (can be adjusted)
    max_penalty = 1     # Max penalty for wrong length
    min_slow_len = 50    # Minimum desired length when mistake exists
    max_slow_len = 512  # Maximum desired length when mistake exists

    for completion, target in zip(completions, ground_truth):
        # Only need target value for length logic
        if isinstance(target, torch.Tensor):
            target = target.item()

        # Length-based adjustment calculation (No base_reward calculation here)
        length_reward_adjustment = 0.0
        completion_tokens = len(tokenizer.encode(completion))

        if target == -1:
            # Penalize long responses for "no mistake" cases
            if completion_tokens > target_fast_len:
                over_limit = completion_tokens - target_fast_len
                penalty = min(max_penalty, (over_limit / (target_fast_len * 2)) * max_penalty) # Gradual penalty
                length_reward_adjustment = -penalty

        else: # target != -1 (mistake exists)
            # Penalize shorter responses for "mistake found" cases
            if completion_tokens < min_slow_len:
                 under_limit = min_slow_len - completion_tokens
                 penalty = min(max_penalty, (under_limit / min_slow_len) * max_penalty)
                 length_reward_adjustment = -penalty
            # Penalize longer respo8nses for "mistake found" cases
            elif completion_tokens > max_slow_len:
                 over_limit = completion_tokens - max_slow_len
                 penalty = min(max_penalty, (over_limit / max_slow_len) * max_penalty) # Gradual penalty
                 length_reward_adjustment = -penalty
            else:
                # Within the desired range [min_slow_len, max_slow_len], no penalty/bonus
                length_reward_adjustment = 0.0

        # Append the adjustment value directly - fixed to use float dtype
        adjustments.append(torch.tensor(length_reward_adjustment, dtype=torch.float32))

    adjustments_tensor = torch.stack(adjustments)

    if len(adjustments) > 0:
        logger.info(f"Length adjustment reward - Mean adjustment: {adjustments_tensor.mean().item():.4f}")

    return adjustments_tensor

@dataclass
class ScriptArguments:
    """Arguments for the GRPO training script."""
    model_name_or_path: str = field(
        default="model-7b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    train_data_dir: str = field(
        default="BIG-Bench-Mistake-Train",
        metadata={"help": "Directory containing training data files"}
    )
    val_data_dir: str = field(
        default="BIG-Bench-Mistake-Test",
        metadata={"help": "Directory containing validation data files"}
    )
    template_path: str = field(
        default="templates/critique_template.txt",
        metadata={"help": "Path to prompt template file"}
    )
    output_dir: str = field(
        default="results/grpo_finetune",
        metadata={"help": "Output directory for model checkpoints"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenizer"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of updates steps to accumulate before backward pass"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max number of training samples to use (for debugging)"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max number of evaluation samples to use (for debugging)"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for training"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X steps"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Linear warmup over this many steps"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load model in 8-bit precision"}
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to load model in 4-bit precision"}
    )
    use_group_rewards: bool = field(
        default=True,
        metadata={"help": "Whether to use group rewards in GRPO"}
    )
    gumbel_samples: int = field(
        default=10, 
        metadata={"help": "Number of Gumbel samples for GRPO"}
    )
    critic_multiple: float = field(
        default=0.5,
        metadata={"help": "Critic loss multiplier"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to deepspeed config file for using deepspeed"}
    )

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    logger.info(f"Loading model {args.model_name_or_path}...")
    
    # Prepare model with quantization if needed
    if args.load_in_8bit:
        quantization_config = {"load_in_8bit": True}
    elif args.load_in_4bit:
        quantization_config = {"load_in_4bit": True, 
                               "bnb_4bit_compute_dtype": torch.float16,
                               "bnb_4bit_quant_type": "nf4"}
    else:
        quantization_config = None
    
    # For deepspeed compatibility, use torch_dtype=None for fp16/bf16 handling by deepspeed
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=None,  # Let DeepSpeed handle the precision
        device_map=None,  # Don't use device_map with DeepSpeed
        quantization_config=quantization_config
    )
    
    # Apply LoRA if specified
    if args.use_lora:
        logger.info("Applying LoRA...")
        if args.load_in_8bit or args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load template
    with open(args.template_path, 'r') as f:
        template = f.read().strip()
    
    # Load training and validation data
    # Override with specific file for overfitting test
    specific_file = "BIG-Bench-Mistake-Train/combined_train.jsonl"
    logger.info(f"Running overfitting test with single file: {specific_file}")
    
    # Load data from the specific file only
    train_data = []
    if os.path.exists(specific_file):
        train_data = load_mistake_data(specific_file)
        logger.info(f"Loaded {len(train_data)} examples for overfitting test")
    else:
        raise FileNotFoundError(f"Specified file not found: {specific_file}")
    
    # Use the same data for validation
    val_data = train_data.copy()
    
    logger.info(f"Using {len(train_data)} examples for both training and validation")
    
    # Create HF datasets
    train_hf_dataset = HFDataset.from_dict({"data": train_data})
    val_hf_dataset = HFDataset.from_dict({"data": val_data})
    
    # Apply preprocessing function
    train_tokenize_func = partial(preprocess_function, tokenizer=tokenizer, template=template, max_length=args.max_length)
    val_tokenize_func = partial(preprocess_function, tokenizer=tokenizer, template=template, max_length=args.max_length)
    
    # Process the datasets
    train_dataset = train_hf_dataset.map(
        train_tokenize_func,
        batched=True,
        remove_columns=["data"],
        desc="Processing training dataset"
    )
    
    val_dataset = val_hf_dataset.map(
        val_tokenize_func,
        batched=True,
        remove_columns=["data"],
        desc="Processing validation dataset"
    )
    
    # Get reward function for base correctness
    base_reward_fn = reward_func
    # Get reward function for length adjustment
    length_adjustment_fn = length_adjustment_reward_func

    # Create a dictionary of reward functions with their weights
    # base_reward_fn handles correctness, length_adjustment_fn handles bonus/penalty for length
    reward_functions = [base_reward_fn, length_adjustment_fn]
    
    # Create training arguments with DeepSpeed compatibility
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_strategy="no",  # Use "steps" instead of "no" for proper evaluation strategy
        eval_steps=args.eval_steps,  # Add this to specify when to evaluate
        save_strategy="steps",
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        save_total_limit=3,
        weight_decay=0.01,
        gradient_checkpointing=True,
        # Let DeepSpeed handle mixed precision (set via config file)
        bf16=True,  
        report_to="wandb",
        max_grad_norm=1.0,
        remove_unused_columns=False,
        use_vllm=True,
        # Generation config
        temperature=0.6,
        num_generations=14,
        # data processings
        max_prompt_length=2048,
        max_completion_length=1024,
        log_completions=True,
        # Remove do_eval parameter as it's redundant with eval_strategy
        # Efficiency improvements from TRL v0.16.0
        scale_rewards=True,  # Enable reward scaling for multiple rewards
        num_iterations=4,     # Enable multi-step optimization (6x faster)
    )
    
    # Create GRPO trainer with multiple reward functions
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add evaluation dataset
        reward_funcs=reward_functions,  # Use multiple reward functions with weights
        callbacks=[SaveBestModelCallback()],  # Add callback to save best model based on rewards
    )
    
    # Train the model
    logger.info("Starting training with DeepSpeed...")
    trainer.train()
    
    # Save the final model - ensure this runs regardless of accelerator
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    logger.info(f"Training completed. Final model saved to {os.path.join(args.output_dir, 'final_model')}")

if __name__ == "__main__":
    main() 
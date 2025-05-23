#!/bin/bash

# Define default parameters
MODEL_PATH="model-7b"
OUTPUT_DIR="results/grpo_finetune"
TRAIN_DIR="BIG-Bench-Mistake-Train"
VAL_DIR="BIG-Bench-Mistake-Test"
TEMPLATE_PATH="templates/critique_template.txt"
BATCH_SIZE=4
GRAD_ACCUM=2
LEARNING_RATE=5e-5
EPOCHS=3
NUM_PROCESSES=8  # Number of GPUs to use
DS_CONFIG="ds_config.json"  # Path to DeepSpeed config file

# Function to display usage information
function display_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run GRPO training on the BIG-Bench-Mistake dataset with DeepSpeed ZeRO Stage 2."
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_PATH     Model name or path (default: $MODEL_PATH)"
    echo "  -o, --output OUTPUT_DIR    Output directory (default: $OUTPUT_DIR)"
    echo "  -b, --batch-size SIZE      Batch size per device (default: $BATCH_SIZE)"
    echo "  -g, --grad-accum STEPS     Gradient accumulation steps (default: $GRAD_ACCUM)"
    echo "  -l, --lr RATE              Learning rate (default: $LEARNING_RATE)"
    echo "  -e, --epochs NUM           Number of training epochs (default: $EPOCHS)"
    echo "  -t, --train-dir DIR        Training data directory (default: $TRAIN_DIR)"
    echo "  -v, --val-dir DIR          Validation data directory (default: $VAL_DIR)"
    echo "  -n, --num-gpus NUM         Number of GPUs to use (default: $NUM_PROCESSES)"
    echo "  -c, --ds-config FILE       DeepSpeed config file path (default: $DS_CONFIG)"
    echo "  --max-train NUM            Maximum number of training samples"
    echo "  --max-eval NUM             Maximum number of evaluation samples"
    echo "  --disable-lora             Disable LoRA (use full fine-tuning)"
    echo "  --disable-4bit             Disable 4-bit quantization"
    echo "  -h, --help                 Display this help message and exit"
}

# Parse command line arguments
POSITIONAL_ARGS=()
MAX_TRAIN=""
MAX_EVAL=""
USE_LORA="--use_lora"
LOAD_4BIT="--load_in_4bit"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            display_help
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -g|--grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -t|--train-dir)
            TRAIN_DIR="$2"
            shift 2
            ;;
        -v|--val-dir)
            VAL_DIR="$2"
            shift 2
            ;;
        -n|--num-gpus)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        -c|--ds-config)
            DS_CONFIG="$2"
            shift 2
            ;;
        --max-train)
            MAX_TRAIN="--max_train_samples $2"
            shift 2
            ;;
        --max-eval)
            MAX_EVAL="--max_eval_samples $2"
            shift 2
            ;;
        --disable-lora)
            USE_LORA="--use_lora False"
            shift
            ;;
        --disable-4bit)
            LOAD_4BIT="--load_in_4bit False"
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Make sure DeepSpeed config exists
if [ ! -f "$DS_CONFIG" ]; then
    echo "Error: DeepSpeed config file '$DS_CONFIG' not found!"
    exit 1
fi

# Display training parameters
echo "Starting GRPO training with DeepSpeed ZeRO Stage 2 and the following parameters:"
echo "  Model: $MODEL_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Training data: $TRAIN_DIR"
echo "  Validation data: $VAL_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation steps: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Number of GPUs: $NUM_PROCESSES"
echo "  DeepSpeed config: $DS_CONFIG"
if [ -n "$MAX_TRAIN" ]; then
    echo "  Max training samples: ${MAX_TRAIN#--max_train_samples }"
fi
if [ -n "$MAX_EVAL" ]; then
    echo "  Max evaluation samples: ${MAX_EVAL#--max_eval_samples }"
fi
echo "  Using LoRA: ${USE_LORA#--use_lora }"
echo "  Using 4-bit quantization: ${LOAD_4BIT#--load_in_4bit }"
echo ""
echo "Starting training in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Run the training script with accelerate
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --machine_rank=0 \
    --num_machines=1 \
    --mixed_precision="bf16" \
    --use_deepspeed \
    --deepspeed_config_file=$DS_CONFIG \
    grpo_train.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_data_dir "$TRAIN_DIR" \
    --val_data_dir "$VAL_DIR" \
    --template_path "$TEMPLATE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$EPOCHS" \
    --deepspeed "$DS_CONFIG" \
    $USE_LORA \
    $LOAD_4BIT \
    $MAX_TRAIN \
    $MAX_EVAL 
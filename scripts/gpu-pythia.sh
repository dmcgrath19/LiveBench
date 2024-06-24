#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N BENCH-PY-1.4
#$ -o /exports/eddie/scratch/s2558433/job_runs/benchPI-1.4_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/benchPI-1.4_$JOB_ID.err
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=500G
#$ -l h_rt=24:00:00
#$ -m bea -M s2558433@ed.ac.uk 

export HF_HOME="/exports/eddie/scratch/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2558433/.cache/conda_pkgs"

export CXXFLAGS="-std=c99"
export CFLAGS="-std=c99"
export TOKENIZERS_PARALLELISM=false

. /etc/profile.d/modules.sh
module unload cuda

module load cuda/12.1.1
#qlogin -q gpu -pe gpu-a100 1 -l h_vmem=500G -l h_rt=24:00:00

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh
module load anaconda

cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/

conda create -n livebench python=3.9
conda activate livebench

pip install causal-conv1d>=1.2.0
pip install mamba-ssm

# python gen_model_answer.py          --bench-name live_bench --model-path /path/to/Mistral-7B-Instruct-v0.2/ --model-id Mistral-7B-Instruct-v0.2 --dtype bfloat16 
# python gen_api_answer.py            --bench-name live_bench --model gpt-4-turbo
# python gen_ground_truth_judgment.py --bench-name live_bench --model-list Mistral-7B-Instruct-v0.2 Llama-2-7b-chat-hf claude-3-opus-20240229
# python show_livebench_results.py    --bench-name live_bench --model-list Mistral-7B-Instruct-v0.2 Llama-2-7b-chat-hf claude-3-opus-20240229


pip install torch packaging
pip install -e .


# Define model identifier
MODEL_ID="EleutherAI/pythia-1.4b"
BENCH_NAME="live_bench"

# Generate model answers
python gen_model_answer.py --model-path $MODEL_ID --model-id $MODEL_ID --bench-name $BENCH_NAME

# Score model outputs
python gen_ground_truth_judgment.py --bench-name $BENCH_NAME

# Display results
python show_livebench_results.py --bench-name $BENCH_NAME

conda deactivate
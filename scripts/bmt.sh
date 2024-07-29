#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N BTLM2-BENCH
#$ -o /exports/eddie/scratch/s2558433/job_runs/bmt_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/bmt_$JOB_ID.err
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

cd /exports/eddie/scratch/s2558433/LiveBench/

# conda create -n livebench python=3.10
conda activate livebench


cd livebench

python gen_model_answer.py --bench-name 'live_bench/instruction_following' --model-path "cerebras/btlm-3b-8k-base" --model-id "btlm-3b-8k-base" --dtype bfloat16 
python gen_model_answer.py --bench-name 'live_bench/reasoning' --model-path "cerebras/btlm-3b-8k-base" --model-id "btlm-3b-8k-base" --dtype bfloat16 



conda deactivate

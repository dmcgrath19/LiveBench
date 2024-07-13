#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N RIF13-BENCH-MAMBA
#$ -o /exports/eddie/scratch/s2558433/job_runs/benchMA-1.4_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/benchMA-1.4_$JOB_ID.err
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

# conda remove env -n livebench

# conda create -n livebench python=3.10
conda activate livebench

# pip install causal-conv1d>=1.4.0
# pip install mamba-ssm

# # python gen_model_answer.py          --bench-name live_bench --model-path /path/to/Mistral-7B-Instruct-v0.2/ --model-id Mistral-7B-Instruct-v0.2 --dtype bfloat16 
# # python gen_api_answer.py            --bench-name live_bench --model gpt-4-turbo
# # python gen_ground_truth_judgment.py --bench-name live_bench --model-list Mistral-7B-Instruct-v0.2 Llama-2-7b-chat-hf claude-3-opus-20240229
# # python show_livebench_results.py    --bench-name live_bench --model-list Mistral-7B-Instruct-v0.2 Llama-2-7b-chat-hf claude-3-opus-20240229

# # pip install torch packaging
# pip install -e .



cd livebench

# # Generate model answers
# python gen_model_answer.py --model-path "EleutherAI/pythia-2.8b" --model-id "pythia-2.8b" --dtype bfloat16 
# python gen_model_answer.py --model-path "EleutherAI/pythia-410m" --model-id "pythia-410m" --dtype bfloat16 
# python gen_model_answer.py --model-path "EleutherAI/pythia-1b" --model-id "pythia-1b" --dtype bfloat16 
# python gen_model_answer.py --model-path "EleutherAI/pythia-160m" --model-id "pythia-160m" --dtype bfloat16 


# python gen_model_answer.py --bench-name 'live_bench/instruction_following' --model-path "state-spaces/mamba-790m-hf" --model-id "mamba-790m-hf" --dtype bfloat16 
# python gen_model_answer.py --bench-name 'live_bench/language' --model-path "state-spaces/mamba-790m-hf" --model-id "mamba-790m-hf" --dtype bfloat16 

# python gen_model_answer.py --bench-name 'live_bench/coding' --model-path "state-spaces/mamba-790m-hf" --model-id "mamba-790m-hf" --dtype bfloat16

# python gen_model_answer.py --bench-name 'live_bench/reasoning' --model-path "state-spaces/mamba-790m-hf" --model-id "mamba-790m-hf" --dtype bfloat16

# python gen_model_answer.py --bench-name 'live_bench/instruction_following' --model-path "state-spaces/mamba-790m-hf" --model-id "mamba-790m-hf" --dtype bfloat16

# python gen_model_answer.py --bench-name 'live_bench/math' --model-path "state-spaces/mamba-790m-hf" --model-id "mamba-790m-hf" --dtype bfloat16 

# python gen_model_answer.py --bench-name 'live_bench/reasoning' --model-path "state-spaces/mamba-1.4b-hf" --model-id "mamba-1.4b-hf" --dtype bfloat16 
#python gen_model_answer.py --bench-name 'live_bench/code' --model-path "state-spaces/mamba-1.4b-hf" --model-id "mamba-1.4b-hf" --dtype bfloat16 


#score the two new ones=
# python gen_ground_truth_judgment.py --bench-name 'live_bench/coding' --model mamba-130m-hf
# python gen_ground_truth_judgment.py --bench-name 'live_bench/language' --model mamba-130m-hf
python gen_ground_truth_judgment.py --bench-name 'live_bench/reasoning' --model mamba-130m-hf
python gen_ground_truth_judgment.py --bench-name 'live_bench/instruction_following' --model mamba-130m-hf
# python gen_ground_truth_judgment.py --bench-name 'live_bench/language' --model mamba-130m-hf
# python gen_ground_truth_judgment.py --bench-name 'live_bench/data_analysis' --model mamba-130m-hf

# #update the .csv files of results
# python show_livebench_results.py

# python gen_ground_truth_judgment.py --bench-name livebench

# python gen_model_answer.py --bench-name 'live_bench/instruction_following' --model-path "state-spaces/mamba-130m-hf" --model-id "mamba-130m-hf" --dtype bfloat16 

# python gen_model_answer.py --model-path "state-spaces/mamba-1.4b-hf" --model-id "mamba-1.4b-hf" --dtype bfloat16 
# python gen_model_answer.py --model-path "state-spaces/mamba-790m" --model-id "mamba-790m" --dtype bfloat16 
# python gen_model_answer.py --model-path "state-spaces/mamba-370m" --model-id "mamba-370m" --dtype bfloat16 
# python gen_model_answer.py --model-path "state-spaces/mamba-130m" --model-id "mamba-130m" --dtype bfloat16 

# # python gen_model_answer.py --model-path $MODEL_ID --model-id $MODEL_ID --bench-name $BENCH_NAME

# # Score model outputs
# # python gen_ground_truth_judgment.py --bench-name livebench
# # Display results
# python download_questions.py

conda deactivate

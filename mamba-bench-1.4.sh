#!/bin/bash

#SBATCH --gres=gpu:a6000 
#SBATCH --time=0-08:00:00
#SBATCH -o job_output.out  # output file for stdout
#SBATCH -e job_error.err  # output file for stderr

set -e
set -x

export HF_HOME="home/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="home/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="home/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="home/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="home/s2558433/.cache/conda_pkgs"

export CXXFLAGS="-std=c99"
export CFLAGS="-std=c99"
export TOKENIZERS_PARALLELISM=false


# source home/s2558433/miniconda3/etc/profile.d/conda.sh
# module load anaconda

# source .bashrc

# cd home/s2558433/LiveBench/

# conda remove env -n livebench

# conda create -n livebench python=3.10

# pip install causal-conv1d>=1.4.0
# pip install mamba-ssm

# # python gen_model_answer.py          --bench-name live_bench --model-path /path/to/Mistral-7B-Instruct-v0.2/ --model-id Mistral-7B-Instruct-v0.2 --dtype bfloat16 
# # python gen_api_answer.py            --bench-name live_bench --model gpt-4-turbo
# # python gen_ground_truth_judgment.py --bench-name live_bench --model-list Mistral-7B-Instruct-v0.2 Llama-2-7b-chat-hf claude-3-opus-20240229
# # python show_livebench_results.py    --bench-name live_bench --model-list Mistral-7B-Instruct-v0.2 Llama-2-7b-chat-hf claude-3-opus-20240229

# pip install torch packaging
pip install -e .



cd livebench

echo "Running the livebench script"

# # Generate model answers

# python gen_model_answer.py --bench-name 'live_bench/reasoning' --model-path "state-spaces/mamba-2.8b-hf" --model-id "mamba-2.8b-hf" --dtype bfloat16 

python gen_model_answer.py --model-path "state-spaces/mamba-1.4b-hf" --model-id "mamba-1.4b-hf" --dtype bfloat16 
# python gen_model_answer.py --model-path "state-spaces/mamba-790m-hf" --model-id "mamba-790m-hf" --dtype bfloat16 
# python gen_model_answer.py --model-path "state-spaces/mamba-370m-hf" --model-id "mamba-370m-hf" --dtype bfloat16 
# python gen_model_answer.py --model-path "state-spaces/mamba-130m-hf" --model-id "mamba-130m-hf" --dtype bfloat16 

# # python gen_model_answer.py --model-path $MODEL_ID --model-id $MODEL_ID --bench-name $BENCH_NAME

# # Score model outputs
# # python gen_ground_truth_judgment.py --bench-name livebench
# # Display results
# python download_questions.py

conda deactivate

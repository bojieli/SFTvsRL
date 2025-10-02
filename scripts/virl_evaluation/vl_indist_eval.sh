#!/bin/bash

#SBATCH --job-name=EvalVI
#SBATCH --mail-user=user@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=1                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=512G
#SBATCH -t 72:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=1   
#SBATCH --output=./slurm_logs/python_array_%A_%a.out
#SBATCH --output=./slurm_logs/python_array_%A_%a.err
#SBATCH --partition=gpu

VITER=2

# enable verification
ENABLE=True
# use absolute action space, consistent with training
ABS=True
NUM_TRAJ=48
CKPT_NAME="tianzhechu/VIRL-VL-Init"
OUTPUT_FOLDER="logs/virl_vl_rule_ood_verify_${VITER}"
PORT=$((RANDOM % 10000 + 2000))

# download from our huggingface dataset repo tianzhechu/SFTvsRL_Data
ROUTE_INFO="YOUR_ROUTE_INFO_PATH" # .json
GPS_TO_PANO="YOUR_GPS_TO_PANO_MAPPING_PATH" # .pkl
STREETVIEWS="YOUR_STREETVIEWS_PATH" # folder of images

DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file scripts/config_zero2_1gpu.yaml --main_process_port ${PORT} \
    -m evaluation.launcher -f evaluation/configs/llama_virl_vl.yaml \
    --model_path=${CKPT_NAME} \
    --output_dir=${OUTPUT_FOLDER}/virl_vl_rule_ood.jsonl \
    --env_config.route_info_path=${ROUTE_INFO} \
    --env_config.platform_cfg.OFFLINE.PANORAMA_DIR=${STREETVIEWS} \
    --env_config.platform_cfg.OFFLINE.GPS_TO_PANO_PATH=${GPS_TO_PANO} \
    --prompt_config.enable_verification=${ENABLE} \
    --env_config.verify_iter=${VITER} \
    --env_config.absolute_action=${ABS} \
    --num_traj=${NUM_TRAJ}

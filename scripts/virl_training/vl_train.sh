#!/bin/bash

#SBATCH --job-name=TrainGP
#SBATCH --mail-user=user@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=8                      # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=1000G
#SBATCH --time=96:00:00                     # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=8   
#SBATCH --output=./slurm_logs/train_%A_%a.out
#SBATCH --output=./slurm_logs/train_%A_%a.err
#SBATCH --partition=gpu

LR=1e-7
save_every=1
save_model=True # one checkpoint ~30GB

CKPT_NAME="tianzhechu/VIRL-VL-Init"
PORT=$((RANDOM % 10000 + 1000))

# download from our huggingface dataset repo tianzhechu/SFTvsRL_Data
BASE_DIR="/root/SFTvsRL_Data/VIRL_routes"
ROUTE_INFO="${BASE_DIR}/nyc_1k_routes/route_infos.json"
GPS_TO_PANO="${BASE_DIR}/nyc_1k_routes/gps_pano_mapping.pkl"
STREETVIEWS="${BASE_DIR}/nyc_1k_routes/street_views/"

DS_SKIP_CUDA_CHECK=1 TOKENIZERS_PARALLELISM=false \
    accelerate launch \
    --config_file scripts/config_zero2_8gpu.yaml \
    --main_process_port ${PORT} -m rl.launcher \
    -f rl/configs/llama_virl_vl.yaml \
    --output_dir=train_ckpt/virl_vl/ \
    --optimizer_config.init_lr=${LR} \
    --optimizer_config.lr_max_steps=20 \
    --prompt_config.enable_verification=True \
    --num_updates=15 \
    --run_name=virl_vl_training \
    --num_steps=256 \
    --model_path=${CKPT_NAME} \
    --save_ckpt=${save_model} \
    --save_every=${save_every} \
    --env_config.route_info_path=${ROUTE_INFO} \
    --env_config.platform_cfg.OFFLINE.PANORAMA_DIR=${STREETVIEWS} \
    --env_config.platform_cfg.OFFLINE.GPS_TO_PANO_PATH=${GPS_TO_PANO} 

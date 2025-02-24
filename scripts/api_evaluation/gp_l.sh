VITER=1
# disable verification
ENABLE=False
# disable rule-ood eval
OOD=False
# choose rule: face card as 10
FACE10=True
# specify target: 24
TARGET=24

NUM_TRAJ=23
MODEL_NAME="" # i.e. gpt-4o
API_URL=""
API_KEY=""
OUTPUT_FOLDER="logs/${MODEL_NAME}_gp_l_indist_verify_${VITER}_target_${TARGET}"

python \
    -m evaluation.launcher -f evaluation/configs/api_gp_language.yaml \
    --output_dir=${OUTPUT_FOLDER}/gp_l_indist.jsonl \
    --model=${MODEL_NAME} \
    --API_URL=${API_URL} \
    --API_KEY=${API_KEY} \
    --prompt_config.enable_verification=${ENABLE} \
    --env_config.target_points=${TARGET} \
    --env_config.verify_iter=${VITER} \
    --env_config.treat_face_cards_as_10=${FACE10} \
    --env_config.ood=${OOD} \
    --num_traj=${NUM_TRAJ}
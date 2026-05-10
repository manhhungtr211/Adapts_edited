DOMAIN=$1
MODEL=$2
ADD_BOS=$3
MODEL_PARALLEL=$4
N_GPU=$5
STOP=$6         # số batch tối đa, để trống = chạy hết (dùng cho failure table)
OUTPUT_DIR='/tmp/output/' # for saving the prediction files
RES_DIR='/tmp/res/' # for saving the evaluation scores of each task
CACHE_DIR='/tmp/cache' # for caching hf models and datasets

# Build chuỗi stop=N chỉ khi STOP có giá trị, tránh lỗi Hydra parse
if [ -n "${STOP}" ]; then
    STOP_ARG="stop=${STOP}"
else
    STOP_ARG=""
fi

if [ ${DOMAIN} == 'biomedicine' ]; then
    TASK='MQP+PubMedQA+RCT+USMLE+ChemProt'
elif [ ${DOMAIN} == 'finance' ]; then
    TASK='ConvFinQA+NER+FPB+FiQA_SA+Headline'
elif [ ${DOMAIN} == 'law' ]; then
    TASK='CaseHOLD+SCOTUS+UNFAIR_ToS'
else
    TASK=${DOMAIN}
fi

echo "Domain-specific tasks: ${TASK}"
echo "MODEL: ${MODEL}"
echo "ADD_BOS: ${ADD_BOS}"
echo "MODEL_PARALLEL: ${MODEL_PARALLEL}"
echo "N_GPU: ${N_GPU}"
echo "STOP: ${STOP} → STOP_ARG='${STOP_ARG}'"   # ← xác nhận stop đã nhận đúng

if [ ${N_GPU} == '8' ]; then
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch  --num_processes ${N_GPU} --multi_gpu \
        inference.py task_name=${TASK} model_name=${MODEL} add_bos_token=${ADD_BOS} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${MODEL_PARALLEL} \
        hydra.run.dir=/tmp ${STOP_ARG}
elif [ ${N_GPU} == '4' ]; then
    CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch  --num_processes ${N_GPU} --multi_gpu \
        inference.py task_name=${TASK} model_name=${MODEL} add_bos_token=${ADD_BOS} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${MODEL_PARALLEL} \
        hydra.run.dir=/tmp ${STOP_ARG}
elif [ ${N_GPU} == '2' ]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES='0,1' accelerate launch --num_processes 1 \
        inference.py task_name=${TASK} model_name=${MODEL} add_bos_token=${ADD_BOS} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${MODEL_PARALLEL} \
        hydra.run.dir=/tmp ${STOP_ARG}
elif [ ${N_GPU} == '1' ]; then
    CUDA_VISIBLE_DEVICES='0' accelerate launch  --num_processes 1 \
        inference.py task_name=${TASK} model_name=${MODEL} add_bos_token=${ADD_BOS} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${MODEL_PARALLEL} \
        hydra.run.dir=/tmp ${STOP_ARG}
fi
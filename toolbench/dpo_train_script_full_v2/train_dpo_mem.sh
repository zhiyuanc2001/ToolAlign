export WORK_DIR="Your work data"
export PYTHONPATH=${WORK_DIR}
LR=1e-6
BETA=0.05

deepspeed --include localhost:1,2,3,4 ${WORK_DIR}/toolbench/dpo_train_script_full_v2/train_dpo_mem.py \
    --model_name_or_path path to sft model  \
    --data_path  path to dpo data \
    --preprocessing_num_workers 1 \
    --conv_template tool-llama-multi-rounds \
    --output_dir your output dir \
    --beta ${BETA} \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --prediction_loss_only \
    --save_strategy epoch \
    --save_total_limit 8 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_steps 5 \
    --lr_scheduler_type linear \
    --logging_steps 2 \
    --seed 42 \
    --bf16 True \
    --tf32 True \
    --disable_tqdm False \
    --source_model_max_length 4096 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --report_to none \
    --deepspeed ${WORK_DIR}/toolbench/dpo_train_script_full_v2/stage3_offload.json

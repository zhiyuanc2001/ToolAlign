export WORK_DIR="Your work dir"
export PYTHONPATH=${WORK_DIR}

deepspeed --num_gpus=4 ${WORK_DIR}/toolbench/sft_train_script_v2/align_train_mem.py \
    --model_name_or_path the path to ToolLLaMA  \
    --data_path  the path to the sft training data \
    --conv_template tool-llama-single-round \
    --output_dir your output data dir \
    --overwrite_output_dir True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy no \
    --prediction_loss_only \
    --save_strategy epoch \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type linear \
    --logging_steps 5 \
    --seed 42 \
    --bf16 True \
    --tf32 True \
    --disable_tqdm False \
    --source_model_max_length 4096 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --deepspeed ${WORK_DIR}/toolbench/sft_train_script_v2/stage3.json

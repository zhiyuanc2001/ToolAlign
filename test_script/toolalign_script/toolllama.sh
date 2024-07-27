export CUDA_VISIBLE_DEVICES=0
export TOOLBENCH_KEY=""
export OPENAI_KEY=""
export OPENAI_API_BASE=""
export PYTHONPATH=./

export GPT_MODEL="gpt-3.5-turbo"
export OUTPUT_DIR="unsafe_output/toolllama"  # output dir
group="unsafe_input_intro"

if [ "$group" == "unsafe_api" ]; then
    unsafe_api=true
else
    unset unsafe_api
fi

mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
python toolbench/inference/qa_pipeline_multithread.py \
    --tool_root_dir server/tools \
    --backbone_model toolllama \
    --model_path your path to ToolLLaMA \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file path to your testset/${group}.json \
    --output_answer_file $OUTPUT_DIR/$group \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1 \
    $([ "$unsafe_api" == "true" ] && echo "--unsafe_api")
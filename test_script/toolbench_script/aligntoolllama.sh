export CUDA_VISIBLE_DEVICES=0
export TOOLBENCH_KEY=""
export OPENAI_KEY=""

export PYTHONPATH=./
export GPT_MODEL="gpt-3.5-turbo"

export METHOD="CoT@1"
export OUTPUT_DIR="toolbench_output/aligntoolllama_sft_${METHOD}"  # your ouptut dir

group=G1_instruction   # # G1_category G1_tool G2_instruction G2_category G3 instruction.

mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
python toolbench/inference/qa_pipeline_multithread.py \
    --tool_root_dir server/tools \
    --backbone_model toolllama \
    --model_path path to AlignToolLLaMA-SFT/DPO \
    --post_training \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method ${METHOD} \
    --input_query_file ../../testset/ToolBench_test/test_instruction/${group}.json \
    --output_answer_file $OUTPUT_DIR/$group \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1
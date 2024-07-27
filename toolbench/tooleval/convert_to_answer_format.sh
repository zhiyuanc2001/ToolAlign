export RAW_ANSWER_PATH=../toolbench_output
export CONVERTED_ANSWER_PATH=../toolbench_output_eval_converted
export MODEL_NAME="aligntoolllama_sft_CoT@1"

export test_set=G1_instruction
mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json

python toolbench/tooleval/convert_to_answer_format.py\
    --answer_dir ${answer_dir} \
    --method CoT@1 \
    --output ${output_file}
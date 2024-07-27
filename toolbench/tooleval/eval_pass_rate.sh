export CONVERTED_ANSWER_PATH=../toolbench_output_eval_converted
export SAVE_PATH=../toolbench_output_eval_result/pass_rate
export CANDIDATE_MODEL="aligntoolllama_sft_CoT@1"
export API_POOL_FILE=your api file/openai_key_7.json

mkdir -p ${SAVE_PATH}
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}


python toolbench/tooleval/eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids ToolAlign/testset/ToolBench_testset/test_query_ids \
    --max_eval_threads 5 \
    --evaluate_times 3

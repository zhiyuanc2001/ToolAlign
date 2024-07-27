export CONVERTED_ANSWER_PATH=../toolbench_output_eval_converted
export SAVE_PATH=../toolbench_output_eval_result/preference_results
export PASS_TARE_PATH=../toolbench_output_eval_result/pass_rate
export REFERENCE_MODEL="chatgpt_CoT@1"

export CANDIDATE_MODEL="aligntoolllama_sft_CoT@1"
export API_POOL_FILE=your api file/openai_key_7.json

mkdir -p ${SAVE_PATH}
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}

python toolbench/tooleval/eval_preference.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${CANDIDATE_MODEL} \
    --test_ids ToolAlign/testset/ToolBench_testset/test_query_ids \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --pass_rate_result_path ${PASS_TARE_PATH} \
    --max_eval_threads 3 \
    --use_pass_rate true \
    --evaluate_times 3
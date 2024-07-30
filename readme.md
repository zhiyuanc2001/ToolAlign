<div align= "center">
    <h1> ToolAlign </h1>
</div>


This is the code for paper **Towards Tool Use Alignment of Large Language Models**.

Our model is on [**Huggingface**](https://huggingface.co/JaxChen/ToolAlign-DPO).


Our data generation process is illustrated in the following image:
![Data Generation Process](asserts/data_generation_v2.png)

## Train Data
The train data are in `trainset` folder

<!-- ToolBench/my_code/extract_tool/toolbench_filtering/G123_train_data_v9.json   and  xueyang  ToolBench/dpo_data/G123_train_data_v3_hf-->

Use `tar -xf dpo_train_data.tar` to unzip the DPO train data.

## Train Script
To obtain AlignToolLLaMA-SFT (SFT on ToolLLaMA), please run `toolbench/sft_train_script_v2/align_train_deepspeed.sh`,


To obtain AlignToolLLaMA-DPO (DPO training on AlignToolLLaMA-DPO), please run `toolbench/dpo_train_script_full_v2/train_dpo_mem.sh`


## Test Data

The testset in ToolAlign for helpfulness, harmlessness, and autonomy are in `testset/`

---
`ToolAlign_testset/`: Testset for harmlessness and autonomy.

`unsafe_api.json`: testset for harmful tool responses.

`unsafe_input_intro.json` and `unsafe_input_safellama.json`: testset for harmful instructions.
1. unsafe_input_intro.json: prompt ChatGPT to transform these instructions into unsafe ones
2. unsafe_input_safellama.json: sample and rewrite from Anthropic Red Teaming Dataset

`without_tool.json`: testset for autonomy.

---

`ToolBench_testset/`: Testset for helpfulness.

This testset is copied from [**ToolBench**](https://github.com/OpenBMB/ToolBench)


## Test Script

First, you should follwing [**ToolBench Data Release**](https://github.com/OpenBMB/ToolBench?tab=readme-ov-file#data-release) to prepare the tool environment.
After downloading the data use Google Drive or Tsinghua Cloud link provided in ToolBench, you can see the `/data/toolenv/tools` folder and please put the folder into `ToolAlign/server` folder.


*We recommend to prepare the tool environment following [**StableToolBench**](https://github.com/THUNLP-MT/StableToolBench#downloading-the-cache) by download the StableToolBench cache to avoid downloading useless data.*  


After downloading the cache provide in StableToolBench, unzip the folder into the `ToolAlign/server` folder and ensure the `ToolAlign/server` folder contains tools folder.


Scripts under `test_script/toolbench_script/` are test for helpfulness, where `aligntoolllama.sh` and `toolllama.sh` are for AlignToolLLaMA-SFT/DPO and ToolLLaMA, respectively.

Scripts under `test_script/toolalign_script/` are test for harmlessness and autonomy, where `aligntoolllama.sh` and `toolllama.sh` are for AlignToolLLaMA-SFT/DPO and ToolLLaMA, respectively.


## Evaluation

For *helpfulness*, we follow [**ToolEval**](https://github.com/OpenBMB/ToolBench?tab=readme-ov-file#tooleval)  in ToolBench to evaluate the LLMs. The following paragraph refers to ToolBench and StableToolBench.
The file structure of the directory should be:
```
├── /aligntoolllama_sft_CoT@1/
│  ├── /G1_instruction/
│  │  ├── /577_CoT@1.json
│  │  └── ...
│  ├── /G1_tool/
│  │  ├── /394_CoT@1.json
│  │  └── ...
│  ├── ...
│  ├── /G3_instruction/
│  │  ├── /455_CoT@1.json
│  │  └── ...
```


You should first prepare the data by run:
`toolbench/tooleval/convert_to_answer_format.sh`

Then you can calculate the pass rate by run `toolbench/tooleval/eval_rass_rate.sh` and calculate the win rate by run `toolbench/tooleval/eval_win_rate.sh`



For *harmlessness* and *autonomy*, the file structure of the directory should be:
The file structure of the directory should be:
```
├── /aligntoolllama_sft/
│  ├── /unsafe_api/
│  │  ├── /517_CoT@1.json
│  │  └── ...
│  ├── /unsafe_input_intro/
│  │  ├── /200_CoT@1.json
│  │  └── ...
│  ├── /unsafe_input_safellama/
│  │  ├── /0_CoT@1.json
│  │  └── ...
│  ├── /without_tool/
│  │  ├── /0_CoT@1.json
│  │  └── ...
```



For *harmlessness*, run `python evaluation/ToolAlign/unsafe_api.py` to evaluate the refusal response rate of harmful tool responses and obtain a .json file.

The structure of the .json file should be
```json
{
  "54905": {
    "is_refuse": true    # or null/false
  },
  ...
}
```
where the key is the test query id.
For harmful tool responses, the refusal response rate is calculated by $\frac{N_{\text{true}}}{N_{\text{true}}+N_{\text{false}}}$, where $N_{\text{true}}$ and $N_{\text{false}}$ are the number of `"is_refuse": true` and the number of `"is_refuse": false` in the saved .json file.


Run `python evaluation/ToolAlign/unsafe_input.py` to evaluate the refusal response rate of harmful instructions and obtain a .json file.
The file structure should be 
```json
{
  "43280": {
    "is_refused": true    # or false
  },
  ...
}
```
where the key is the test query/response id.  
We found that for responses not refused due to being harmful but because the tool could not resolve the issue after being called, GPT-4 often categorizes them as successfully refusing unsafe instructions.
**So, for harmful instructions, we manually check the the refusal responses with** `"is_refused": true`, and label the *fake* refusal response as `"is_refused": false`.  
Finally, you can calculate the refusal response rate by $\frac{N_{\text{true}}}{N_{\text{all}}}$, where $N_{\text{true}}$ is the number of `"is_refused": true` after manual check in the .json file and $N_{\text{all}}$ is the number of all unsafe instructions. (attention: our unsafe instruction testset include `unsafe_input_intro.json` and `unsafe_input_safellama.json`).


For autonomy, run `python evaluation/ToolAlign/without_tool.py` to evaluate the direct response rate, and you will obtain a .json file.
The file structure should be 
```json
{
  "47": {
    "answer_without_tool": false    # or true
  },
  ...
}
```

You can calculate the direct response rate by 
$\frac{N_{\text{true}}}{N_{\text{all}}}$, where $N_{\text{true}}$ is the number `"answer_without_tool": true` in the .json file, and $N_{\text{all}}$ is the number of test samples.

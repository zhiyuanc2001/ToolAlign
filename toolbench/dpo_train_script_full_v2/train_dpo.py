# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import pathlib
from typing import Dict, Optional
import torch
import transformers
from full_dpo_trainer import DPOTrainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers import set_seed
from toolbench.tool_conversation import SeparatorStyle
from toolbench.model.model_adapter import get_conversation_template
from toolbench.train.llama_condense_monkey_patch import replace_llama_with_condense
from datasets import load_from_disk


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
torch.set_printoptions(profile="full")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    conv_template: str = field(
        default=None, metadata={"help": "Template used to format the training data."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."},
    )
    remove_unused_columns: bool = field(default=False)
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": ("The loss type for DPO.")})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    source_model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Original maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def preprocess(conv, roles, messages):
    conv.messages = []
    for sentence in messages:
        role = roles[sentence["from"]]
        conv.append_message(role, sentence["value"])
    return conv.get_prompt()


def apply_chat_template(example, template):
    # example: [prompt chosen rejected]
    conv = get_conversation_template(template)
    if template == "tool-llama":
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
        roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    chosen_messages, rejected_messages = example["chosen"], example["rejected"]

    example["text_prompt"] = example["prompt"]
    example["text_chosen"] = preprocess(conv, roles, chosen_messages)
    example["text_rejected"] = preprocess(conv, roles, rejected_messages)

    return example
    


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.source_model_max_length < training_args.model_max_length:
        condense_ratio = int(training_args.model_max_length/training_args.source_model_max_length)
        # ratio = N means the sequence length is expanded by N, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
        print(f"==condense ratio: {condense_ratio}===")
        replace_llama_with_condense(ratio=condense_ratio)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    ###############
    # Load Datasets
    ###############
    raw_datasets = load_from_disk(data_args.data_path)
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "template": data_args.conv_template
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["prompt", "chosen", "rejected", "query_id"],
        desc="Formatting comparisons with conversation template",
    )
    raw_datasets["train"] = raw_datasets["train"].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        cache_dir=training_args.cache_dir
    )

    model = model_args.model_name_or_path
    ref_model = model
    ref_model_kwargs = model_kwargs


    #########################
    # Instantiate DPo Trainer
    #########################
    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        loss_type=training_args.loss_type,
        peft_config=None
    )
    

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        train_result = trainer.train()
        
    # save checkpoint 
    trainer.save_model(output_dir=training_args.output_dir)

    trainer.save_state()
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    print("*** Training complete :-) ***")
    


if __name__ == "__main__":
    train()

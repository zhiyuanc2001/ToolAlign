# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from toolbench.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from toolbench.dpo_train_script_full_v2.train_dpo import train

if __name__ == "__main__":
    train()

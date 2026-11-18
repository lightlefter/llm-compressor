"""
REAP Expert Pruning Example with Token Masking (Moonlight)

This example demonstrates REAP pruning with token masking, so that only
assistant responses contribute to the expert saliency statistics
``S_j = mean(g_j * ||f_j||_2)``. User prompts and role headers are excluded:
the pruning decision reflects only the tokens a deployment would generate.

The calibration flow is the same as in `reap_moonlight_16b.py`; only the
loss_mask field and `use_loss_mask=True` are added.
"""

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier

# Select model and load it.
# Note: for the Moonlight model, you may need to install the
# `tiktoken` package for the tokenizer, and the tokenizer itself is
# implemented via custom code (tokenization_moonshot), so loading it
# non-interactively requires trust_remote_code=True.
model_id = "moonshotai/Moonlight-16B-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Get special token IDs for the masking logic below.
# Moonlight chat format:
#   <|im_assistant|>assistant<|im_middle|>[content]<|im_end|>
# Each message is prefixed with a role token and the literal role word, then
# the content follows the <|im_middle|> separator until <|im_end|>.
im_assistant_id = tokenizer.convert_tokens_to_ids("<|im_assistant|>")
im_middle_id = tokenizer.convert_tokens_to_ids("<|im_middle|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

# Select calibration dataset.
# open-perfectblend is a multi-turn conversation dataset (the dataset behind
# the "perfectblend" shortcut used by reap_moonlight_16b.py).
DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train"
ROLE_MAP = {"human": "user", "gpt": "assistant"}

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess(example):
    messages = [
        {"role": ROLE_MAP.get(msg["from"], msg["from"]), "content": msg["value"]}
        for msg in example["conversations"]
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


ds = ds.map(preprocess)


def tokenize(sample):
    """
    Tokenize inputs and create loss_mask for assistant responses.

    The mask marks tokens that are part of assistant responses with 1, and all
    other tokens (user messages, role headers) with 0. This makes REAP
    accumulate its expert saliency statistics over assistant outputs only.

    Moonlight chat format:
        <|im_assistant|>assistant<|im_middle|>[content]<|im_end|>

    Unlike Llama-3 (see the GPTQ masking example), the chat template does not
    prepend a BOS token, so there is no attention-sink token to keep and the
    mask covers exactly the assistant content.
    """
    tokenized = tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

    # Create mask: 1 for tokens in assistant responses, 0 otherwise
    input_ids = tokenized["input_ids"]
    mask = [0] * len(input_ids)

    # Find all assistant response segments
    i = 0
    while i < len(input_ids):
        if input_ids[i] == im_assistant_id:
            i += 1
            # Skip the literal "assistant" role word up to <|im_middle|>
            while i < len(input_ids) and input_ids[i] != im_middle_id:
                i += 1
            i += 1  # skip <|im_middle|>
            # Mark content tokens until <|im_end|>
            while i < len(input_ids) and input_ids[i] != im_end_id:
                mask[i] = 1
                i += 1
        else:
            i += 1

    tokenized["loss_mask"] = torch.tensor(mask)
    return tokenized


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Prune 25% of the least salient experts in each MoE layer (64 -> 48 experts),
# where saliency is accumulated over assistant responses only. Per-expert
# statistics are written to moonlight_report_masked.json along the way.
recipe = REAPPruningModifier(sparsity=0.25, report_path="moonlight_report_masked.json")

# Apply algorithms with token masking enabled.
# use_loss_mask=True tells REAP to accumulate the expert saliency only over
# tokens where loss_mask=1.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    use_loss_mask=True,
    moe_calibrate_all_experts=False,  # Disable calibrating all experts for REAP
)

# Save to disk compressed.
# This is done before the generation check below: on GPUs that cannot hold
# the full model, dispatch_model offloads modules and generation may exhaust
# host memory and take the process down (the OOM killer cannot be caught),
# in which case the calibration results must already be on disk.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-REAP-25-masked"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# Confirm generations of the compressed model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

"""
REAP Expert Pruning Example with Token Masking

This example demonstrates REAP pruning with token masking, so that only
assistant responses contribute to the expert saliency statistics
``S_j = mean(g_j * ||f_j||_2)``. User prompts and role headers are excluded:
the pruning decision reflects only the tokens a deployment would generate.

The calibration flow is the same as in `qwen3_example.py`; only the loss_mask
field and `use_loss_mask=True` are added.
"""

import torch
from compressed_tensors.offload import init_dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.utils import load_context

# Select model and load it.
init_dist()
model_id = "Qwen/Qwen3-30B-A3B-Thinking-2507"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="disk", offload_folder="offload_folder"
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Get special token IDs for the masking logic below.
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
assistant_id = tokenizer.convert_tokens_to_ids("assistant")

# Select calibration dataset.
# open-perfectblend is a multi-turn conversation dataset (the dataset behind
# the "perfectblend" shortcut used by qwen3_example.py).
DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train"
ROLE_MAP = {"human": "user", "gpt": "assistant"}

NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}])


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

    Qwen chat format:
        <|im_start|>assistant\n[content]<|im_end|>

    For thinking models the template inserts a <think>...</think> section at
    the start of the assistant response; it is part of the generated response
    and is masked in as well.
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
    # (<|im_start|> immediately followed by the "assistant" role token)
    i = 0
    while i < len(input_ids) - 1:
        if input_ids[i] == im_start_id and input_ids[i + 1] == assistant_id:
            i += 2  # skip role header tokens
            # Mark content tokens until <|im_end|>
            while i < len(input_ids) and input_ids[i] != im_end_id:
                mask[i] = 1
                i += 1
        else:
            i += 1

    tokenized["loss_mask"] = torch.tensor(mask)
    return tokenized


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Prune the 25% least salient experts from every layer (128 -> 96 experts),
# where saliency is accumulated over assistant responses only. Per-expert
# statistics are written to qwen3_report_masked.json along the way.
recipe = REAPPruningModifier(sparsity=0.25, report_path="qwen3_report_masked.json")

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
    sequential_targets_per_subgraph=4,
    batch_size=16,
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-REAP-25-masked"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

"""Write the Kaggle fine-tuning notebook (robot/train/kaggle_sft.ipynb) from the cells below.

It follows Liquid's own recipe for LFM3-VL-2B (cookbook lfm3_vl_sft_with_trl.ipynb): the same library versions, LoRA
on the language model only (vision encoder and projector untouched, so the original F16 mmproj keeps working), loss on
the assistant's answer only. Run `python -m robot.train.notebook` after changing a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

INTRO = """# Sprocket planner: fine-tune LFM3-VL-2B on simulator-verified programs (SFT + LoRA)

Follows Liquid's recipe for this model: same library versions, LoRA on the language model only (the vision encoder and
projector stay untouched, so the original F16 mmproj keeps working), loss on the assistant's answer only.

**Before running:** add the dataset holding `train.jsonl` and `valid.jsonl` (Add data, top right), and set the
accelerator to a GPU (T4 or P100) with internet on. Run once with `SMOKE = True`: 10 steps on 64 examples check the
whole pipeline in about 15 minutes. Then set `SMOKE = False` for the real run (a few hours on a T4).

**Outputs** in `/kaggle/working`: `sprocket-lora.gguf`, an adapter for llama-server `--lora` on top of the QAD Q4_0
model, and `sprocket-planner-Q4_0.gguf`, the merged model. Both are scored on the 50-task suite at home."""

SETTINGS = """SMOKE = True        # True: 10 steps on 64 examples to check the pipeline; False: the real run
DATA = "/kaggle/input/sprocket-sft"   # the dataset you added; change it if you named it differently
EPOCHS = 2"""

INSTALL = ('!pip install -qqq "torch==2.14.1" "torchvision==0.29.1" "torchao==0.18.0" "trl==1.16.0" '
           '"transformers==5.20.2" "peft==0.21.2" "accelerate==1.15.0" "datasets==5.1.0" "huggingface-hub==2.3.0" '
           '"tokenizers==0.23.3" "safetensors==0.8.0" --progress-bar off')

SETUP = """import json
import os

import torch

os.environ["WANDB_DISABLED"] = "true"
use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
dtype = torch.bfloat16 if use_bf16 else torch.float16  # a T4 has no bfloat16: float16, as in Liquid's notebook
print("GPU:", torch.cuda.get_device_name(0), "| precision:", dtype)"""

LOAD = """from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "LiquidAI/LFM3-VL-2B"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map="auto", trust_remote_code=True)
print(f"{model.num_parameters():,} parameters")"""

DATA_MD = """## The data

Each line is one conversation in the planner's own format: the system prompt with the robot's command card, the room
as the robot saw it plus the instruction, and the answer (`say:`, `plan:`, a Python program). Every answer ran in the
simulator and passed its task's check before it was kept. The robot runs the model with thinking off; the chat
template then writes an empty `<think></think>` before the answer, in training and at use alike."""

DATA_CODE = """from datasets import Dataset


def load(name):
    rows = [json.loads(line) for line in open(os.path.join(DATA, name), encoding="utf-8")]
    return [{"messages": [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                          for m in r["messages"]]} for r in rows]

train_rows, valid_rows = load("train.jsonl"), load("valid.jsonl")
if SMOKE:
    train_rows, valid_rows = train_rows[:64], valid_rows[:8]
train_dataset, valid_dataset = Dataset.from_list(train_rows), Dataset.from_list(valid_rows)
print(len(train_dataset), "train,", len(valid_dataset), "valid")
print(processor.apply_chat_template(train_rows[0]["messages"], tokenize=False, enable_thinking=False)[-600:])
lengths = [len(processor.apply_chat_template(r["messages"], tokenize=True, enable_thinking=False))
           for r in train_rows[:200]]
print("tokens per conversation: max", max(lengths), "mean", sum(lengths) // len(lengths))"""

TRAIN = """from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=r".*language_model.*\\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    task_type="CAUSAL_LM",
)
sft_config = SFTConfig(
    output_dir="./sprocket-sft",
    num_train_epochs=EPOCHS,
    max_steps=10 if SMOKE else -1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.05,
    max_length=4096,
    assistant_only_loss=True,
    gradient_checkpointing=True,
    logging_steps=5 if SMOKE else 25,
    eval_strategy="no" if SMOKE else "steps",
    eval_steps=200,
    save_strategy="no",
    bf16=use_bf16,
    fp16=not use_bf16,
    report_to="none",
)
trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_dataset,
                     eval_dataset=None if SMOKE else valid_dataset, processing_class=processor,
                     peft_config=peft_config)
trainer.model.print_trainable_parameters()
trainer.train()
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")"""

LOOK_MD = """## A quick look at what the fine-tuned planner writes

Greedy answers for a few held-out instructions. The real score comes from the 50-task suite at home."""

LOOK = """trainer.model.eval()
for row in valid_rows[:4]:
    messages = row["messages"][:2]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False).to(trainer.model.device)
    with torch.inference_mode():
        out = trainer.model.generate(**inputs, max_new_tokens=300, do_sample=False)
    print("INSTRUCTION:", messages[1]["content"][0]["text"].split("Instruction:")[-1].strip())
    print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    print("-" * 60)"""

SAVE = """trainer.save_model("./sprocket-lora")
merged = trainer.model.merge_and_unload()
merged.save_pretrained("./sprocket-merged")
processor.save_pretrained("./sprocket-merged")
print(sorted(os.listdir("./sprocket-merged")))"""

GGUF_MD = """## GGUF for llama.cpp

The merged model becomes a Q4_0 GGUF (the format that runs at full speed on the robot's NPU), and the adapter becomes
a GGUF LoRA that llama-server can apply on top of the QAD Q4_0 model with `--lora`."""

GGUF = """!git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
!pip install -qqq -r /kaggle/working/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt --progress-bar off
!python /kaggle/working/llama.cpp/convert_hf_to_gguf.py ./sprocket-merged --outfile sprocket-planner-F16.gguf --outtype f16
!python /kaggle/working/llama.cpp/convert_lora_to_gguf.py ./sprocket-lora --base-model-id LiquidAI/LFM3-VL-2B --outfile sprocket-lora.gguf
!cmake -S /kaggle/working/llama.cpp -B /kaggle/working/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null
!cmake --build /kaggle/working/llama.cpp/build --target llama-quantize -j 4 > /dev/null
!/kaggle/working/llama.cpp/build/bin/llama-quantize sprocket-planner-F16.gguf sprocket-planner-Q4_0.gguf Q4_0
!rm -f sprocket-planner-F16.gguf && ls -lh *.gguf"""

CELLS = [("markdown", INTRO), ("code", SETTINGS), ("code", INSTALL), ("code", SETUP), ("code", LOAD),
         ("markdown", DATA_MD), ("code", DATA_CODE), ("code", TRAIN), ("markdown", LOOK_MD), ("code", LOOK),
         ("code", SAVE), ("markdown", GGUF_MD), ("code", GGUF)]


def main() -> None:
    cells = []
    for kind, text in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}}
    path = Path(__file__).with_name("kaggle_sft.ipynb")
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()

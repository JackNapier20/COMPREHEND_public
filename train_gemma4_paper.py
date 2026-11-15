#!/usr/bin/env python3
"""Retrain Gemma 4 on the original paper extraction request and smoke-test it."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from statistics import mean
from unittest.mock import patch

from gemma_prompts import gemma_main_idea_messages

ROOT = Path(__file__).resolve().parent
USER_MARKER = "<|turn>user\n"
MODEL_MARKER = "<|turn>model\n"
TURN_END = "<turn|>"
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"
V6_SYSTEM = (
    "You are a professional editor. Your task is to extract the single most important "
    "main idea from a given text in one short sentence."
)
V6_SUFFIX = (
    "\n\nExtract the single most important main idea in one short sentence.\n"
    "Return only that sentence."
)
EXTRACTION_FORMAT = "original-paper-single-item-v1"

CONFIG = {
    "model_name": "/gypsum/work1/zamani/sivaraaman/models/gemma-4-31B-it",
    "max_seq_length": 4608,
    "load_in_4bit": True,
    "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0,
    "train_test_split": 0.2,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    "warmup_steps": 5, "num_train_epochs": 4, "max_steps": -1,
    "learning_rate": 2e-4, "logging_steps": 10, "optim": "adamw_8bit",
    "weight_decay": 0.01, "lr_scheduler_type": "linear", "seed": 3407,
    "eval_steps": 50, "save_steps": 50,
}


def read_dataset(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8-sig")
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(records, dict):
        records = next((records[k] for k in ("train", "data", "examples")
                        if isinstance(records.get(k), list)), [records])
    if not isinstance(records, list) or not records:
        raise ValueError("Training file is empty")
    converted = []
    counts = Counter()
    for index, row in enumerate(records):
        messages = row.get("messages", [])
        if (len(messages) != 3
                or [message.get("role") for message in messages] != ["system", "user", "assistant"]
                or any(not isinstance(message.get("content"), str) or not message["content"].strip()
                       for message in messages)):
            raise ValueError(f"Row {index}: expected nonempty system/user/assistant messages")
        system, user, target = messages
        if system["content"] == V6_SYSTEM:
            text = user["content"]
            if not text.startswith("Text:\n") or not text.endswith(V6_SUFFIX):
                raise ValueError(f"Row {index}: not the verified V6 extraction format")
            text = text[len("Text:\n"):-len(V6_SUFFIX)]
            if not text.strip():
                raise ValueError(f"Row {index}: empty extraction input")
            converted.append({"messages": [*gemma_main_idea_messages(text), target]})
            counts["main_idea"] += 1
        else:
            if "main idea" in system["content"].lower():
                raise ValueError(f"Row {index}: unrecognized extraction system prompt")
            converted.append({"messages": messages})
            counts["summary"] += 1
    if not all(counts[task] for task in ("main_idea", "summary")):
        raise ValueError("Expected both V6 extraction and summary records")
    if len({json.dumps(row, sort_keys=True) for row in converted}) != len(converted):
        raise ValueError("Exact duplicate records in the training file")
    return converted


def text_tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def render_training_text(messages: list[dict], processor) -> tuple[str, str]:
    prefix = processor.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    if not full.startswith(prefix):
        history = processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        head = history + MODEL_MARKER
        # Google's 31B template primes an empty thought channel only during inference.
        # Include that same prefix in SFT, as recommended for no-thinking datasets.
        if full.startswith(head) and prefix == head + EMPTY_THOUGHT:
            full = prefix + full[len(head):]
        else:
            raise ValueError("Training and inference template prefixes disagree")
    if (full.count(USER_MARKER) != 1 or full.count(MODEL_MARKER) != 1
            or "<|turn>system\n" not in prefix
            or not full.endswith(TURN_END + "\n")):
        raise ValueError("Expected a single Gemma 4 system/user/assistant conversation")
    if full[len(prefix):] != messages[-1]["content"].strip() + TURN_END + "\n":
        raise ValueError("Chat template modified the assistant target")
    return full, prefix


def prepare_records(records: list[dict], processor, max_length: int) -> tuple[list[dict], dict, dict]:
    tokenizer = text_tokenizer(processor)
    prepared, expectations, lengths, too_long = [], {}, [], []
    for index, row in enumerate(records):
        full, prefix = render_training_text(row["messages"], processor)
        encoded = tokenizer(full, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        if (tokenizer.bos_token_id is None or ids[0] != tokenizer.bos_token_id
                or ids.count(tokenizer.bos_token_id) != 1):
            raise ValueError(f"Row {index}: expected exactly one BOS token")
        lengths.append(len(ids))
        if len(ids) > max_length:
            too_long.append({"row_index": index, "tokens": len(ids)})
        model_boundary = full.index(MODEL_MARKER) + len(MODEL_MARKER)
        target_end = len(full) - len(TURN_END + "\n")
        expectation = {
            "row_index": index,
            "prompt_positions": [i for i, (_, end) in enumerate(offsets) if end <= model_boundary],
            "target_positions": [i for i, (start, end) in enumerate(offsets)
                                 if end > len(prefix) and start < target_end],
            "eos_position": max(i for i, token in enumerate(ids)
                                if token == tokenizer.convert_tokens_to_ids(TURN_END)),
        }
        if not expectation["target_positions"]:
            raise ValueError(f"Row {index}: no assistant target tokens")
        key = tuple(ids)
        if key in expectations:
            raise ValueError(f"Rows {expectations[key]['row_index']} and {index} tokenize identically")
        expectations[key] = expectation
        prepared.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    if too_long:
        raise ValueError(
            f"{len(too_long)} examples exceed max_seq_length={max_length}; maximum={max(lengths)}. "
            f"Increase --max-seq-length; no examples were truncated or dropped. Rows: {too_long}"
        )
    audit = {"records": len(records), "min_tokens": min(lengths), "max_tokens": max(lengths),
             "mean_tokens": mean(lengths), "max_seq_length": max_length,
             "overlength_records": 0, "truncated_or_dropped": 0,
             "thinking": False, "chat_template_sha256": hashlib.sha256(
                 processor.chat_template.encode("utf-8")).hexdigest()}
    return prepared, expectations, audit


def validate_masks(dataset, expectations: dict) -> dict:
    fractions, original_rows = [], []
    for row in dataset:
        ids = [int(token) for token in row["input_ids"]]
        labels = [int(label) for label in row["labels"]]
        expected = expectations.get(tuple(ids))
        if expected is None or len(ids) != len(labels):
            raise ValueError("Trainer changed or truncated a tokenized example")
        if any(label not in (-100, token) for label, token in zip(labels, ids)):
            raise ValueError("Response mask contains invalid labels")
        if (any(labels[i] != -100 for i in expected["prompt_positions"])
                or any(labels[i] == -100 for i in expected["target_positions"])
                or labels[expected["eos_position"]] == -100):
            raise ValueError(f"Row {expected['row_index']}: broken prompt/answer/EOS loss mask")
        fractions.append(sum(label != -100 for label in labels) / len(labels))
        original_rows.append(expected["row_index"])
    if not fractions:
        raise ValueError("Training/validation split is empty after masking")
    return {"records": len(fractions), "min_label_fraction": min(fractions),
            "mean_label_fraction": mean(fractions), "original_row_indices": original_rows}


def response_mask_arguments(tokenizer) -> dict:
    arguments = {"tokenizer": tokenizer, "num_proc": 1}
    instruction = getattr(tokenizer, "_unsloth_input_part", None)
    response = getattr(tokenizer, "_unsloth_output_part", None)
    if instruction is not None or response is not None:
        if (not isinstance(instruction, str) or not instruction.startswith(USER_MARKER)
                or not isinstance(response, str) or not response.startswith(MODEL_MARKER)):
            raise ValueError("Loaded Unsloth response markers are not Gemma 4-compatible")
        # Recent Unsloth refuses explicit markers if the loader already configured them.
        return arguments
    return {**arguments, "instruction_part": USER_MARKER, "response_part": MODEL_MARKER}


def extraction_smoke_cases(records):
    from dataset_prep.sample_wb_random import normalize

    source = ROOT / "comparison_256_outputs/WB_current_k2/evaltree_capability_base.json"
    nodes = json.loads(source.read_text(encoding="utf-8"))["levels"][0]["nodes"]
    by_id = {node["id"]: node["prompt"] for node in nodes}
    inputs = [normalize(row["messages"][1]["content"]) for row in records]
    cases = []
    for leaf_id in (137, 169):
        text = by_id[leaf_id]
        if any(normalize(text) in candidate for candidate in inputs):
            raise ValueError(f"Smoke-test WB256 leaf {leaf_id} appears in training inputs")
        cases.append((f"main_idea_wb256_{leaf_id}", gemma_main_idea_messages(text), 256))
    return cases


def pair_smoke_case():
    from common_summary_generation import summarize
    from common_summary_generation.prompt_variants import CURRENT_VARIANT
    from tree import Level, Node, Tree

    tasks = ("Write Python code to import a CSV of bus arrivals.",
             "Write Python code to filter a CSV of train departures by station.")
    tree = Tree(levels=[Level(level=0, nodes=[
        Node(id=0, prompt=tasks[0], match=1, score=0.9, alive=True),
        Node(id=1, prompt=tasks[1], match=0, score=0.9, alive=False),
    ])], next_id=2)
    with patch.dict(os.environ, {"BLOSSOM_SUMMARY_PROMPT_VARIANT": CURRENT_VARIANT}), \
            patch.object(summarize, "gemma_chat_text", return_value="Process transit CSVs.") as call:
        summarize.generate_summary(tree, 0, threshold=-1.0, model_name="gemma",
                                   api_keys={"gemma": {"concurrency": 1}})
        if call.call_count != 1:
            raise RuntimeError("Expected one production pair-summary request")
        return ("comprehend_pair_summary", call.call_args.kwargs["messages"], 1024)


def smoke_generate(model, processor, cases, torch, results) -> None:
    tokenizer = text_tokenizer(processor)
    turn_end = tokenizer.convert_tokens_to_ids(TURN_END)
    for task, messages, cap in cases:
        prefix = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=cap, do_sample=False, use_cache=True,
                eos_token_id=turn_end,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        ended = bool(generated and generated[-1] == turn_end)
        results.append({"task": task, "response": text, "output_tokens": len(generated),
                        "ended_at_turn_end": ended})
        print(f"{task} smoke response: {text}", flush=True)
        if not text or not ended:
            raise RuntimeError(f"{task} smoke generation was empty or did not reach the turn-end token")
        if task.startswith("main_idea") and ("\n" in text or text.startswith(("```", "{", "["))):
            raise RuntimeError(f"{task} smoke generation was not a plain single-line main idea")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=CONFIG["model_name"])
    hpc = Path("/gypsum/work1/zamani/sivaraaman")
    parser.add_argument("--dataset", type=Path, default=hpc / "finetuneDatasetv6.json")
    parser.add_argument("--output-dir", type=Path, default=hpc / "outputs/gemma-4-31b-a40-paper-prompt")
    parser.add_argument("--save-dir", type=Path, default=hpc / "gemma-4-31b-a40-merged-bf16-paper-prompt")
    parser.add_argument("--max-seq-length", type=int, default=CONFIG["max_seq_length"])
    parser.add_argument("--preflight-only", action="store_true", help="Load only tokenizer, validate all rows; no GPU, model weights or W&B")
    parser.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--resume", nargs="?", const=True, default=None, help="Latest checkpoint or explicit checkpoint path")
    parser.add_argument("--skip-merge", action="store_true", help="Save adapters only; merging also needs substantial host RAM/disk")
    parser.add_argument("--skip-post-merge-check", action="store_true", help="Skip reloading the saved merged weights for a 4-bit smoke check")
    args = parser.parse_args(argv)
    if args.max_seq_length < 1:
        parser.error("--max-seq-length must be positive")
    if args.resume and args.preflight_only:
        parser.error("--resume cannot be combined with --preflight-only")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    records = read_dataset(args.dataset)
    smoke_cases = extraction_smoke_cases(records)
    if args.preflight_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        _, _, audit = prepare_records(records, tokenizer, args.max_seq_length)
        print(json.dumps(audit, indent=2))
        print("Preflight passed; no model weights loaded and no training started.")
        return

    # Unsloth must patch the training stack before Torch/Transformers/TRL imports.
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a CUDA GPU with BF16 support")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("Training output directory is nonempty; use --resume or a new --output-dir")
    if not args.skip_merge and args.save_dir.exists() and any(args.save_dir.iterdir()):
        raise FileExistsError("Merged model directory is nonempty; use a new --save-dir")
    config = {**CONFIG, "model_name": args.model_name, "max_seq_length": args.max_seq_length,
              "dataset": str(args.dataset.resolve()), "output_dir": str(args.output_dir.resolve()),
              "save_dir": str(args.save_dir.resolve()), "dtype": "bfloat16",
              "extraction_format": EXTRACTION_FORMAT}
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    report_path = args.output_dir / "training_manifest.json"
    if args.resume:
        if not report_path.exists():
            raise ValueError("Resume requires the matching paper-prompt training manifest")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if (previous["dataset_sha256"] != dataset_hash
                or any(previous["config"].get(key) != config[key] for key in (
                    "model_name", "max_seq_length", "seed", "train_test_split", "lora_r", "lora_alpha",
                    "extraction_format",
                ))):
            raise ValueError("Resume dataset, model, split or LoRA configuration differs from the original run")
    os.environ.setdefault("WANDB_PROJECT", "gemma-finetune")
    os.environ.setdefault("WANDB_ENTITY", "jacknapier20-umass-amherst")
    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
    print(f"Loading Gemma 4: {args.model_name}", flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=args.model_name, max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16, load_in_4bit=True, full_finetuning=False,
    )
    if not str(getattr(model.config, "model_type", "")).startswith("gemma4"):
        raise ValueError("Expected a Gemma 4 checkpoint, not a renamed Gemma 3 model")
    tokenizer = text_tokenizer(processor)
    tokenizer.eos_token = TURN_END
    model.generation_config.eos_token_id = tokenizer.convert_tokens_to_ids(TURN_END)
    prepared, expectations, audit = prepare_records(records, processor, args.max_seq_length)
    print("All examples fit without truncation:", json.dumps(audit), flush=True)
    print("Validation is a row-level SFT loss monitor, not independent tree/benchmark evaluation.", flush=True)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=CONFIG["seed"],
        use_rslora=False, loftq_config=None,
    )
    split = Dataset.from_list(prepared).train_test_split(
        test_size=CONFIG["train_test_split"], seed=CONFIG["seed"],
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir), per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=1, gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"], num_train_epochs=CONFIG["num_train_epochs"],
        max_steps=CONFIG["max_steps"], learning_rate=CONFIG["learning_rate"], fp16=False, bf16=True,
        logging_steps=CONFIG["logging_steps"], optim=CONFIG["optim"], weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"], seed=CONFIG["seed"],
        eval_strategy="steps", eval_steps=CONFIG["eval_steps"], prediction_loss_only=True,
        save_strategy="steps", save_steps=CONFIG["save_steps"], save_total_limit=3,
        report_to=[] if args.report_to == "none" else ["wandb"], run_name="gemma-4-31b-a40-V5",
        max_length=args.max_seq_length, packing=False, dataset_num_proc=1,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=split["train"],
                         eval_dataset=split["test"], args=training_args)
    trainer = train_on_responses_only(trainer, **response_mask_arguments(tokenizer))
    mask_audit = {"train": validate_masks(trainer.train_dataset, expectations),
                  "validation": validate_masks(trainer.eval_dataset, expectations)}
    if sum(info["records"] for info in mask_audit.values()) != len(records):
        raise ValueError("Response masking silently removed training/validation examples")
    smoke_cases.append(pair_smoke_case())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("unsloth", "unsloth_zoo", "torch", "transformers", "trl", "datasets", "peft", "bitsandbytes"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    run_report = {"config": config, "dataset_sha256": dataset_hash,
                  "tokenization": audit, "response_masks": mask_audit, "packages": packages,
                  "task_counts": dict(Counter("main_idea" if row["messages"][1]["content"].startswith(
                      "You will be given a text. Extract the single most important main idea")
                                             else "summary" for row in records)),
                  "validation_scope": "Row-level SFT loss, correlated within one teacher tree"}
    report_path.write_text(json.dumps(run_report, indent=2) + "\n", encoding="utf-8")
    run = None
    if args.report_to == "wandb":
        import wandb
        run = wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ.get("WANDB_ENTITY"),
                         name="gemma-4-31b-a40-paper-prompt", config=config)
    try:
        print("STARTING TRAINING", flush=True)
        trainer.train(resume_from_checkpoint=args.resume)
        run_report["final_validation_loss"] = trainer.evaluate()
        adapter_dir = args.output_dir / "lora_adapters"
        model.save_pretrained(str(adapter_dir))
        processor.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        if args.skip_merge or args.skip_post_merge_check:
            FastModel.for_inference(model)
            run_report["adapter_smoke"] = []
            smoke_generate(model, processor, smoke_cases, torch, run_report["adapter_smoke"])
        if not args.skip_merge:
            print(f"Saving merged 16-bit weights: {args.save_dir}", flush=True)
            model.save_pretrained_merged(str(args.save_dir), processor, save_method="merged_16bit")
            processor.save_pretrained(str(args.save_dir))
            tokenizer.save_pretrained(str(args.save_dir))
            if not args.skip_post_merge_check:
                del trainer, model
                gc.collect()
                torch.cuda.empty_cache()
                # Reload the saved merged checkpoint in 4-bit to fit the training GPU.
                merged, merged_processor = FastModel.from_pretrained(
                    model_name=str(args.save_dir), max_seq_length=args.max_seq_length,
                    dtype=torch.bfloat16, load_in_4bit=True, full_finetuning=False,
                )
                FastModel.for_inference(merged)
                run_report["post_merge_smoke_4bit_reload"] = []
                smoke_generate(merged, merged_processor, smoke_cases, torch,
                               run_report["post_merge_smoke_4bit_reload"])
        run_report["status"] = "complete"
        print(f"Training complete; adapters: {adapter_dir}; report: {report_path}", flush=True)
    except Exception as error:
        run_report["status"] = "failed"
        run_report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path.write_text(json.dumps(run_report, indent=2) + "\n", encoding="utf-8")
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()

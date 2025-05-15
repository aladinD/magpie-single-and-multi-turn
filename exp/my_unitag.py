#!/usr/bin/env python3
import os
import json
import warnings
import concurrent.futures
import argparse
import re

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset

from utils import (
    load_jsonl_to_list,
    load_dataset_from_file,
    save_dataset,
    make_api_request_with_retry,
    get_conversation_template,
)
from str_utils import (
    input_difficulty_rating,
    input_classification,
    input_quality_rating,
)
from lingua import LanguageDetectorBuilder

import sys
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Unified Tagging Manager.")
    parser.add_argument("--tag_mission", type=str, default="quality",
        choices=["difficulty", "quality", "classification", "llm_classification", "safety", "reward", "language", "llm_reward"],
        help="The tagging mission."
    )
    parser.add_argument("--model_path", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct", help="Tag Model."
    )
    parser.add_argument("--guard_model_path", type=str,
        default="meta-llama/Meta-Llama-Guard-2-8B", help="Guard Model."
    )
    parser.add_argument("--reward_model_path", type=str,
        default="sfairXC/FsfairX-LLaMA3-RM-v0.1", help="Reward Model."
    )
    parser.add_argument("--input_file", type=str, required=True,
        help="Input dataset file name"
    )
    parser.add_argument("--batch_size", type=int, default=1000,
        help="Number of samples per batch."
    )
    parser.add_argument("--checkpoint_every", type=int, default=2,
        help="Save checkpoint every n batches"
    )
    parser.add_argument("--api", action="store_true", help="Use API to generate responses")
    parser.add_argument("--debug", action="store_true", help="Debug mode: only first 100 samples")
    parser.add_argument("--save_as", type=str, default="json", choices=["json", "jsonl"],
        help="Save the generated responses as JSON or JSONL"
    )

    # vLLM / generation configs
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16"])
    parser.add_argument("--quantization", type=str, default="fp8",
        choices=["fp8","awq","gptq","None"]
    )
    parser.add_argument("--kv_cache_dtype", type=str, default="auto", choices=["auto","fp8"])
    parser.add_argument("--max_model_len", type=int, default=131072)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)

    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    return parser.parse_args()


args = get_args()
print(f"[unitag.py] Unified Tagging Manager. Arguments: {args}")

MODEL_NAME = args.model_path
mission = args.tag_mission
batch_size = args.batch_size
checkpoint_every = args.checkpoint_every if mission != "reward" else args.checkpoint_every * 100

# API config
if MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
    api_model_name = "meta-llama/Llama-3-8b-chat-hf"
elif MODEL_NAME == "meta-llama/Meta-Llama-3-70B-Instruct":
    api_model_name = "meta-llama/Llama-3-70b-chat-hf"
else:
    api_model_name = MODEL_NAME

API_ENDPOINT = args.api_url if hasattr(args, "api_url") else None
API_HEADERS = {"Authorization": args.api_key} if hasattr(args, "api_key") and args.api_key else {}
API_PARAMS = {
    "model": api_model_name,
    "max_tokens": args.max_tokens,
    "temperature": args.temperature,
    "repetition_penalty": args.repetition_penalty,
    "stop": ["}"],
}


# === New helper to stringify conversations ===

def format_conversation(conv_list):
    """
    Convert a list of {from:,value:} dicts into a single string.
    """
    lines = []
    for turn in conv_list:
        speaker = turn.get("from", "unknown")
        text = turn.get("value", "").replace("\n", " ")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def input_llm_reward(str_conversation):
    user_message = f'''
## Instruction 

You will be given a conversation between a User and an AI assistant.

Please rate the assistant's response to each instruction. In particular, when judging, consider:

- Instruction Following: Does the response directly address the question?
- Accuracy: Is the information provided in the response correct?
- Presentation: Is the response logically structured and easy to understand?

## Conversation
```
{str_conversation}
```

## Output Format
Please respond **exactly** with a JSON object of this shape (no extra braces, no shorthand):
```json
{{
  "score": <an integer between 0 and 5>,
  "explanation": "<a non-empty string explaining your rating>"
}}
```

For instance:
```json
{{
 "score": 4,
 "explanation": "The assistant understood the question, provided correct facts, but its structure could be clearer ..."
}}
```

Make sure to adhere to this formatting.
'''
    return user_message


def input_llm_classification(str_conversation):
    user_message = f'''
# Instruction

You will be given a conversation between a User and an AI assistant.

You will need to tag the conversation/task accordingly.

## Conversation
```
{str_conversation}
```

## Tagging the Conversation
Please analyze the conversation and select the most relevant task tag from the list below:

all_task_tags = [
    "Information seeking",  # Users ask for specific information or facts about various topics.
    "Reasoning",  # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",  # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",  # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",  # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",  # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
    "Data analysis",  # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",  # Users seek assistance with crafting stories, poems, or other creative texts. 
    "Advice seeking",  # Users ask for recommendations or guidance on various personal or professional issues.
    "Brainstorming",  # Involves generating ideas, creative thinking, or exploring possibilities. 
    "Others"  # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]

## Output Format:
You can only select a single primary tag. Other tags can be added to the list of "other_tags".
Please respond **exactly** with a JSON object of this shape (no extra braces, no shorthand):
```json
{{ 
    "primary_tag": "<primary tag>",
    "other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```

For instance:
```json
{{
    "primary_tag": "Information seeking",
    "other_tags": ["Advice seeking", "Others"]
}}
```

Make sure to adhere to this formatting.
'''
    return user_message


def template_generator(text: str, mission: str) -> str:
    if mission == "difficulty":
        return input_difficulty_rating(text)
    elif mission == "quality":
        return input_quality_rating(text)
    elif mission == "classification":
        return input_classification(text)
    elif mission == "llm_classification":
        return input_llm_classification(text)
    elif mission == "llm_reward":
        return input_llm_reward(text)
    else:
        raise ValueError(f"Invalid mission: {mission}")


# ---- JSON sanitization ----

_KEYS = ["intent", "knowledge", "difficulty"]

def sanitize_json(raw: str) -> str:
    # normalize key casing
    for key in _KEYS:
        raw = re.sub(rf'"{key}"\s*:', f'"{key}":', raw, flags=re.IGNORECASE)
    # insert missing commas
    raw = re.sub(r'"\s*\n\s*"', '",\n    "', raw)
    # close odd quotes/braces
    if raw.count('"') % 2 == 1:
        raw = raw.rstrip()
        if not raw.endswith('"'): raw += '"'
        if not raw.endswith('}'): raw += '}'
    return raw

def parse_tags(raw: str) -> dict:
    fixed = sanitize_json(raw)
    data = json.loads(fixed, strict=False)
    return {k.lower(): v for k, v in data.items()}

# keys we expect in the full JSON cases
_JSON_KEYS = {
    "difficulty":    ["intent","knowledge","difficulty"],
    "quality":       ["input_quality","explanation"],
    "classification":["primary_tag","other_tags"],
    "llm_classification":["primary_tag","other_tags"],
    "llm_reward":    ["score","explanation"],
}

def normalize_braces(raw: str) -> str:
    """
    Collapse any duplication of outer {{ … }} to { … } 
    and strip any leading/trailing junk outside the first {...}.
    """
    raw = raw.strip()
    # collapse leading '{{' to '{'
    raw = re.sub(r'^\{\s*\{', '{', raw)
    # collapse trailing '}}' to '}'
    raw = re.sub(r'\}\s*\}$', '}', raw)
    # extract only the first {...} block
    m = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    return m[0] if m else raw

def escape_backslashes(raw: str) -> str:
    """
    JSON requires \\ for every single backslash.
    We'll replace any \ that isn't already \\ with \\.
    """
    # first, collapse any existing "\\\\" to a placeholder
    raw = raw.replace('\\\\', '<BACKSLASH>')
    # then escape single backslashes
    raw = raw.replace('\\', '\\\\')
    # restore our placeholder
    return raw.replace('<BACKSLASH>', '\\\\')

def sanitize_and_parse(raw: str, mission: str) -> dict:
    raw = normalize_braces(raw)
    # handle bare-number "{4}" shorthand for llm_reward
    if mission == "llm_reward" and re.fullmatch(r'\{\s*([0-5])\s*\}', raw):
        digit = int(re.search(r'\d', raw).group())
        return {"score": digit, "explanation": ""}

    # escape stray backslashes so JSON loads
    raw = escape_backslashes(raw)

    # ensure commas between lines if missing
    # (only applies if we expect multiple fields)
    raw = re.sub(r'"\s*\n\s*"', '",\n    "', raw)

    # finally, load
    data = json.loads(raw, strict=False)
    # lowercase keys
    data = {k.lower():v for k,v in data.items()}
    return data

def process_engine_responses(response: str, item: dict, mission: str) -> dict:
    try:
        tags = sanitize_and_parse(response, mission)
        if mission in ("difficulty","quality","classification","llm_classification","llm_reward"):
            for key in _JSON_KEYS[mission]:
                if key not in tags:
                    raise KeyError(f"Missing key {key}")
        # now assign back to item:
        if mission == "difficulty":
            item.update({
                "intent": tags["intent"],
                "knowledge": tags["knowledge"],
                "difficulty": tags["difficulty"],
                "difficulty_generator": MODEL_NAME
            })
        elif mission == "quality":
            item.update({
                "input_quality": tags["input_quality"],
                "quality_explanation": tags["explanation"],
                "quality_generator": MODEL_NAME
            })
        elif mission == "classification":
            item.update({
                "task_category": tags["primary_tag"],
                "other_task_category": tags["other_tags"],
                "task_category_generator": MODEL_NAME
            })
        elif mission == "llm_classification":
            item.update({
                "llm_task_category": tags["primary_tag"],
                "llm_other_task_category": tags["other_tags"],
                "llm_task_category_generator": MODEL_NAME
            })
        elif mission == "llm_reward":
            item.update({
                "llm_reward": tags["score"],
                "llm_reward_explanation": tags["explanation"]
            })
    except Exception as e:
        print(f"[unitag.py] Failed to process item: {e}")
        print(f"[unitag.py] Raw response: {response}")
        # reset for that mission
        reset_keys = []
        if mission in _JSON_KEYS:
            for k in _JSON_KEYS[mission]:
                # map JSON key → item field
                if mission == "quality" and k=="explanation":
                    reset_keys += ["quality_explanation","quality_generator"]
                elif mission == "llm_reward" and k=="explanation":
                    reset_keys += ["llm_reward_explanation"]
                else:
                    reset_keys.append(k)
            # dedupe
            reset_keys = list(set(reset_keys))
        for k in reset_keys:
            item[k] = None
    return item

# Process a batch of data using the API
def process_batch_with_api(batch, mission):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_item = {}
        for item in batch:
            if mission in ("llm_reward", "llm_classification"):
                convo_str = format_conversation(item['conversations'])
                user_content = ( input_llm_reward(convo_str)
                          if mission=="llm_reward"
                          else input_llm_classification(convo_str) )
            else:
                user_content = template_generator(item['instruction'], mission)
            future = executor.submit(
                make_api_request_with_retry,
                [
                    {'role':'user',    'content': user_content},
                    {'role':'assistant','content': "{"}
                ],
                API_PARAMS, API_ENDPOINT, API_HEADERS
            )
            future_to_item[future] = item

        for fut in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[fut]
            resp = fut.result()
            resp = "{" + resp + "}"
            process_engine_responses(resp, item, mission)
    return batch


def process_batch(batch, llm, params, mission, tokenizer=None):
    prompts = []
    for item in batch:
        if mission == "safety":
            # Safety: only user+assistant
            chat = [
                {"role": "user",      "content": item["instruction"]},
                {"role": "assistant", "content": item["response"]},
            ]
            template = tokenizer.apply_chat_template(chat, tokenize=False)

        elif mission in ("llm_reward", "llm_classification"):
            # Full conversation for both llm_reward and llm_classification
            convo_str = format_conversation(item["conversations"])
            if mission == "llm_reward":
                system_msg = input_llm_reward(convo_str)
            else:  # llm_classification
                system_msg = input_llm_classification(convo_str)

            conv = get_conversation_template(MODEL_NAME)
            conv.append_message(conv.roles[0], system_msg)
            conv.append_message(conv.roles[1], None)
            template = conv.get_prompt()
            # note: our classification/reward prompts include their own braces,
            # so we don’t append “{” here

        else:
            # difficulty / quality / vanilla classification
            conv = get_conversation_template(MODEL_NAME)
            conv.append_message(
                conv.roles[0],
                template_generator(item["instruction"], mission)
            )
            conv.append_message(conv.roles[1], None)
            template = conv.get_prompt() + "{"

        prompts.append(template)

    # generate all at once
    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        cell = outputs[i]

        # 1️⃣ guard against no candidates
        if not getattr(cell, "outputs", None) or len(cell.outputs) == 0:
            print(f"[unitag.py] Warning: no outputs for index {i}, marking missing.")
            # set mission‐specific fields to None
            if mission == "safety":
                item["llama_guard_2"] = None
            elif mission == "llm_reward":
                item.update({"llm_reward": None, "llm_reward_explanation": None})
            elif mission == "llm_classification":
                item.update({
                    "llm_task_category": None,
                    "llm_other_task_category": None,
                    "llm_task_category_generator": None
                })
            else:
                pass
            continue

        # 2️⃣ normal flow
        raw_text = cell.outputs[0].text.strip()

        # strip code fences / stray braces
        model_response = re.sub(r'^```(?:json)?\s*|```$', "", raw_text, flags=re.MULTILINE).strip()
        # truncate after last “}”
        if "}" in model_response:
            model_response = model_response[:model_response.rfind("}")+1]
        # drop everything before first “{”
        if "{" in model_response:
            model_response = model_response[model_response.find("{"):]
        # ensure leading brace
        if not model_response.startswith("{"):
            model_response = "{" + model_response

        # dispatch to your JSON parser/fallback
        item = process_engine_responses(model_response, item, mission)

    return batch


def process_batch_with_reward_model(batch, rm_pipe, rm_pipe_kwargs):
    prompts = []
    for i, item in enumerate(batch):
        input = item['instruction']
        output = item['response']
        chat = [
            {"role": "user", "content": f"{input}"},
            {"role": "assistant", "content": f"{output}"},
        ]
        template = rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")
        prompts.append(template)
    
    outputs = rm_pipe(prompts, **rm_pipe_kwargs)
    scores = [output[0]["score"] for output in outputs]

    for i, item in enumerate(batch):
        try:
            item['instruct_reward'] = scores[i]
            item['reward_model'] = args.reward_model_path
        except Exception as e:
            print(f"Failed to process item: {item} with error: {str(e)}")

            item['instruct_reward'] = None
            item['reward_model'] = args.reward_model_path
    return batch

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, mission, llm, params, api, rm_pipe, rm_pipe_kwargs, batch_size, checkpoint_file, checkpoint_every = 20):
    if os.path.exists(checkpoint_file):
        last_checkpoint_idx = len(load_dataset_from_file(checkpoint_file))
        print(f"[unitag.py] Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_file)
        num_batches = (len(dataset) - last_checkpoint_idx + batch_size - 1) // batch_size
        print(f"[unitag.py] Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate total number of batches
        print(f"[unitag.py] Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size + last_checkpoint_idx
        end_idx = min((i + 1) * batch_size + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]

        if api:
            batch = process_batch_with_api(batch, mission)
        elif mission == "reward":
            batch = process_batch_with_reward_model(batch, rm_pipe, rm_pipe_kwargs)
        elif mission == "safety":
            tokenizer = AutoTokenizer.from_pretrained(args.guard_model_path)
            batch = process_batch(batch, llm, params, mission, tokenizer)
        else:
            batch = process_batch(batch, llm, params, mission)

        dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file every checkpoint_every batches
        if (i + 1) % checkpoint_every == 0:
            save_dataset(dataset[:end_idx], checkpoint_file)
            print(f"[unitag.py] Dataset checkpoint saved after batch {i + 1}.")

    return dataset

if __name__ == "__main__":
    input_file = args.input_file
    # set output & checkpoint per mission
    base, ext = os.path.splitext(input_file)
    if mission == "difficulty":
        output_file     = f"{base}_difficulty.jsonl"
        checkpoint_file = f"{base}_difficulty_checkpoint.json"
    elif mission == "quality":
        output_file     = f"{base}_quality.jsonl"
        checkpoint_file = f"{base}_quality_checkpoint.json"
    elif mission == "classification":
        output_file     = f"{base}_category.jsonl"
        checkpoint_file = f"{base}_category_checkpoint.json"
    elif mission == "llm_classification":
        output_file     = f"{base}_llm_category.jsonl"
        checkpoint_file = f"{base}_llm_category_checkpoint.json"
    elif mission == "safety":
        output_file     = f"{base}_safety.jsonl"
        checkpoint_file = f"{base}_safety_checkpoint.json"
    elif mission == "reward":
        output_file     = f"{base}_reward.jsonl"
        checkpoint_file = f"{base}_reward_checkpoint.json"
    elif mission == "llm_reward":
        output_file     = f"{base}_llm_reward.jsonl"
        checkpoint_file = f"{base}_llm_reward_checkpoint.json"
    elif mission == "language":
        output_file     = f"{base}_language.jsonl"
        checkpoint_file = f"{base}_language_checkpoint.json"
    else:
        raise ValueError(f"Unknown mission: {mission}")

    if args.save_as == "json":
        output_file = output_file.replace(".jsonl", ".json")

    # load dataset
    if not args.debug:
        dataset = load_dataset_from_file(input_file)
    else:
        warnings.warn("Debug mode: only first 100 samples")
        dataset = load_dataset_from_file(input_file)[:100]

    # dispatch
    if mission != "language":
        # prepare engine
        if args.api and mission in ["difficulty", "quality", "classification", "llm_classification", "llm_reward"]:
            llm, params, rm_pipe = None, None, None
        else:
            if mission in ["difficulty","quality","classification","llm_classification","llm_reward", "safety"]:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.device
                llm = LLM(
                    model = MODEL_NAME if mission != "safety" else args.guard_model_path,
                    dtype = args.dtype,
                    quantization = None if args.quantization=="None" else args.quantization,
                    kv_cache_dtype = args.kv_cache_dtype,
                    max_model_len = args.max_model_len,
                    tensor_parallel_size = args.tensor_parallel_size,
                    gpu_memory_utilization = args.gpu_memory_utilization,
                    trust_remote_code = True,
                    enable_prefix_caching = True,
                )
                params = SamplingParams(
                    temperature = args.temperature,
                    max_tokens = args.max_tokens,
                    repetition_penalty = args.repetition_penalty,
                    stop = ["}"],
                    include_stop_str_in_output = True,
                )
                rm_pipe, rm_pipe_kwargs = None, None
            # handle safety & reward as original...
        # run the generation & tagging
        updated = generate_and_update(
            dataset, mission, llm, params, args.api,
            rm_pipe, rm_pipe_kwargs,
            batch_size, checkpoint_file, checkpoint_every
        )
    else:
        # language detection
        detector = LanguageDetectorBuilder.from_all_languages().build()
        for item in dataset:
            try:
                item['language'] = detector.detect_language_of(item['instruction']).iso_code_639_1.name
            except:
                item['language'] = None
        updated = dataset

    # save final
    if args.save_as == "json":
        save_dataset(updated, output_file, convert_to_jsonl=False)
    else:
        save_dataset(updated, output_file, convert_to_jsonl=True)

    # cleanup
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("[unitag.py] Finished and removed checkpoint.")
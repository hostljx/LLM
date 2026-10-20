import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "/mnt/workspace/model/qwen2.5-7B-Instruct"
DEFAULT_LORA_PATH = "/mnt/workspace/model/qwen2.5-7B-LoRA"
DEFAULT_SYSTEM_PROMPT = (
    "您是一位在临床推理、诊断和治疗计划方面具有高级知识的医学专家。"
    "请回答以下医学问题。"
)


def check_model_path(base_model_path: str, lora_model_path: str) -> None:
    """Fail early with a useful message when inference paths are incorrect."""
    base_path = Path(base_model_path)
    lora_path = Path(lora_model_path)

    is_huggingface_id = (
        not base_path.is_absolute()
        and len(base_path.parts) == 2
        and not base_model_path.startswith((".", "~"))
    )
    if not is_huggingface_id and not base_path.is_dir():
        raise FileNotFoundError(
            f"基础模型路径不存在: {base_model_path}\n"
            "请确认服务器上已下载基础模型，并把 --base-model 指向包含 config.json 的目录；"
            "如果使用 Hugging Face 模型，请传入类似 Qwen/Qwen2.5-7B-Instruct 的模型名。"
        )

    if not lora_path.exists():
        raise FileNotFoundError(
            f"LoRA 路径不存在: {lora_model_path}\n"
            "请把 --lora-path 改为训练后包含 adapter_config.json 的目录。"
        )

    required_files = ("adapter_config.json", "adapter_model.safetensors")
    missing_files = [
        file_name
        for file_name in required_files
        if not (lora_path / file_name).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"LoRA 目录缺少文件: {', '.join(missing_files)}\n"
            f"当前目录: {lora_model_path}\n"
            "如果实际保存的是 checkpoint，请把 --lora-path 指向具体 checkpoint 目录。"
        )


def load_model(base_model_path: str, lora_model_path: str):
    check_model_path(base_model_path, lora_model_path)

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA。Qwen2.5-7B 推理建议在服务器 GPU 上运行。")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_model_path)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_answer(
    question: str,
    tokenizer,
    model,
    max_new_tokens: int = 1024,
    show_think: bool = True,
) -> tuple[str, str]:
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if show_think:
        system_prompt += (
            "请先使用 <think> 和 </think> 标签输出简要分析过程，"
            "然后再输出最终回答。"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    full_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    visible_answer = full_answer
    if "</think>" in visible_answer:
        visible_answer = visible_answer.split("</think>", 1)[1].strip()

    if not show_think and full_answer != visible_answer:
        return full_answer, visible_answer
    return full_answer, full_answer


def save_result(
    output_path: str,
    question: str,
    full_answer: str,
    visible_answer: str,
) -> None:
    result = {
        "question": question,
        "answer": full_answer,
        "answer_without_think": visible_answer,
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Qwen2.5-7B LoRA 进行医学问答推理")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--question", help="直接推理一个问题；不提供时进入交互模式")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--show-think", action="store_true")
    parser.add_argument("--save-json", help="将最后一次推理结果保存为 JSON 文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.base_model, args.lora_path)

    if args.question:
        questions = [args.question]
    else:
        print("模型加载完成，输入医学问题开始推理；输入 exit 或 quit 退出。")
        questions = None

    while True:
        question = questions.pop(0) if questions else input("\n问题：").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            print("问题不能为空。")
            continue

        full_answer, visible_answer = generate_answer(
            question,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            show_think=args.show_think,
        )
        print("\n完整输出：")
        print(full_answer if args.show_think else visible_answer)

        if args.save_json:
            save_result(args.save_json, question, full_answer, visible_answer)
            print(f"\n结果已保存到: {args.save_json}")

        if questions is not None:
            break


if __name__ == "__main__":
    main()
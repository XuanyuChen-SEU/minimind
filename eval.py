"""
MokioMind 推理与对话脚本 (eval.py)

在项目中的角色：
  - 与 train_pretrain.py / train_full_sft.py 并列，作为「训练完成后」的推理入口。
  - 从 save_dir 加载 .pth 权重（如 full_sft_512.pth），或从 HuggingFace 路径加载；不依赖 MokioMind 文件夹。
  - 使用 model.model 中的 MokioMindConfig、MokioMindForCausalLM；tokenizer 来自 load_from（默认 model 目录）。

数据流与张量形状概览：
  1. 输入：用户/系统文本 → tokenizer.apply_chat_template(conversation) → 字符串
  2. 编码：字符串 → tokenizer(..., return_tensors="pt") → input_ids [batch=1, seq_len], attention_mask [1, seq_len]
  3. 生成：model.generate(input_ids, attention_mask, ...) → generated_ids [1, seq_len + new_tokens]
  4. 解码：generated_ids[0][prompt_len:] → tokenizer.decode(...) → 仅“新生成”的回复字符串

核心逻辑：
  - 对话模式：多轮 conversation 列表 + apply_chat_template 拼成一条带角色标记的序列再 tokenize
  - 预训练模式：仅 BOS + 原始 prompt 字符串，无对话模板
  - 历史截断：conversation[-historys:] 控制携带的最近轮数（需偶数）
  - 流式输出：TextStreamer 在每生成一个 token 时回调解码并打印
"""
import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model import MokioMindConfig, MokioMindForCausalLM
# from model.model_lora import apply_lora, load_lora  # ！修正：原缺少LoRA加载支持
from trainer.trainer_utils import setup_seed

warnings.filterwarnings("ignore")


def init_model(args):
    """加载 tokenizer 与 MokioMind/HuggingFace 模型，并可选加载 LoRA。

    模型前向时的张量形状（供参考）：
      - input_ids: [batch_size, seq_len]，整型 token id
      - attention_mask: [batch_size, seq_len]，1=有效位置，0=padding
      - 内部：embed_tokens(input_ids) → hidden [bsz, seq_len, hidden_size]
      - 输出 logits: [bsz, seq_len, vocab_size]（或 generate 时只保留最后一步）
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if "model" in args.load_from:
        # 使用项目内 MokioMind 配置与结构（非 HuggingFace 仓库）
        model = MokioMindForCausalLM(
            MokioMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(
                    args.use_moe
                ),  # ！修正：原缺少use_moe参数，MoE模型无法正确加载
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )
        moe_suffix = "_moe" if hasattr(args, "use_moe") and args.use_moe else ""
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        # state_dict 中 key 与 model.state_dict() 完全一致时 strict=True 才通过
        model.load_state_dict(
            torch.load(ckp, map_location=args.device), strict=True
        )  # ！修正：原strict=False会静默忽略丢失/多余的权重键

        # ！修正：原缺少LoRA加载逻辑（需取消顶部 apply_lora/load_lora 注释）
        if args.lora_weight != "None":
            apply_lora(model)
            load_lora(
                model,
                f"./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth",
            )
    else:
        # 从 HuggingFace 路径加载（如 "meta-llama/Llama-2-7b"），含 config + 权重
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        )
    print(
        f"MokioMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)"  # ！修正：原残留MiniMind命名
    )
    return model.eval().to(args.device), tokenizer


def main():
    # ---------- 参数：模型来源、结构、生成与对话 ----------
    # 以下 save_dir / weight / hidden_size / use_moe 需与训练时保存的路径和配置一致（如 train_full_sft 默认保存到 save_dir/full_sft_512.pth）
    parser = argparse.ArgumentParser(
        description="MokioMind模型推理与对话"
    )
    parser.add_argument(
        "--load_from",
        default="model",
        type=str,
        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录（与训练脚本的 --save_dir 对应）")
    parser.add_argument(
        "--weight",
        default="full_sft",
        type=str,
        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo），对应训练保存的文件名前缀",
    )
    parser.add_argument(
        "--lora_weight",
        default="None",
        type=str,
        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）",
    )
    parser.add_argument(
        "--hidden_size",
        default=512,
        type=int,
        help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=8,
        type=int,
        help="隐藏层数量（Small/MoE=8, Base=16）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--inference_rope_scaling",
        default=False,
        action="store_true",
        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=8192,
        type=int,
        help="最大生成长度（注意：并非模型实际长文本能力）",
    )
    parser.add_argument(
        "--temperature",
        default=0.85,
        type=float,
        help="生成温度，控制随机性（0-1，越大越随机）",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）"
    )
    parser.add_argument(
        "--historys",
        default=0,
        type=int,
        help="携带历史对话轮数（需为偶数，0表示不携带历史）",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="运行设备",
    )
    args = parser.parse_args()

    # 自动测试时的固定问题列表；手动模式下一行用 input() 读
    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]

    conversation = []  # 多轮对话列表，每项 {"role":"user"|"assistant", "content": str}
    model, tokenizer = init_model(args)
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    # 流式输出：每生成一个 token 就解码并打印，skip_prompt 不重复打印输入
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("👶: "), "")
    for prompt in prompt_iter:
        setup_seed(2026)  # 固定种子便于复现；可改为 random.randint(0, 2048) 增加随机性
        if input_mode == 0:
            print(f"👶: {prompt}")
        # 历史截断：只保留最近 historys 条消息（需为偶数，保证 user/assistant 成对）
        conversation = conversation[-args.historys :] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 构造模型输入字符串：对话模式用 chat 模板；预训练模式仅 BOS + 原始 prompt
        templates = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if args.weight == "reason":
            templates["enable_thinking"] = True  # 仅Reason模型使用
        inputs = (
            tokenizer.apply_chat_template(**templates)
            if args.weight != "pretrain"
            else (tokenizer.bos_token + prompt)
        )
        # 编码为张量。形状：input_ids [1, L], attention_mask [1, L]，L = 当前序列长度
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print("🤖️: ", end="")
        # generate 输入/输出形状：
        #   input_ids:      [1, L]       → 与 attention_mask 一致
        #   generated_ids:  [1, L + N]   N = 实际生成 token 数（≤ max_new_tokens，遇 EOS 提前停）
        # 内部每步：模型输出 logits [1, 1, vocab_size]（或 [1, prefix_len, vocab_size]），
        # 经 top_p/temperature 采样得到下一个 token，拼到序列后，再作为下一步输入（或仅用 last token + KV cache）
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )
        # 只解码“新生成”部分：generated_ids[0] 形状 [L+N]，去掉前 L 个 prompt token
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": response})
        print("\n\n")


if __name__ == "__main__":
    main()  # 交互/自动测试循环：构造 conversation → tokenize → generate → decode → 写回 conversation

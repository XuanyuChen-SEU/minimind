"""
minimind 项目入口 (main.py)

项目概览：
  - 本仓库实现 MokioMind 小规模因果语言模型（类 LLaMA 结构），支持预训练、SFT、推理与可选 LoRA。
  - 目录结构：
      model/        模型定义（MokioMindConfig、Attention、MoE、MokioMindForCausalLM 等）与 LoRA
      dataset/      数据封装（PretrainDataset、SFTDataset、DPODataset、RLAIFDataset）
      trainer/      训练脚本（train_pretrain.py、train_full_sft.py）与 trainer_utils
      eval.py       推理与对话脚本（加载权重、generate、流式输出）
  - 典型使用方式：
      预训练：  python -m trainer.train_pretrain --data_path ../dataset/xxx.jsonl [--from_weight none]
      SFT：     python -m trainer.train_full_sft --from_weight pretrain --data_path ../dataset/xxx.jsonl
      推理：    python eval.py --load_from model --weight full_sft [--hidden_size 512]
  - 注意：MokioMind 文件夹为独立 demo，与本项目代码无耦合，请勿与本目录下的 model/ 混淆。
"""


def main():
    print("Hello from minimind!")


if __name__ == "__main__":
    main()

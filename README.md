---
license: other
language:
- zh
tags:
- clinical-pathways
- medical-nlp
- tokenization
- safetensors
---

# 临床路径词元与参数样例

本数据集基于卫生健康主管部门发布的临床路径编制，提供 5 个病种的结构化词元序列和参数张量，可用于临床路径建模、路径遵循评测和参数读取验证。

| 病种 | 路径版本 | ICD-10 | 序列词元数 |
| --- | --- | --- | ---: |
| 急性 ST 段抬高心肌梗死 | 2009 版 | I21.0–I21.3 | 100 |
| 慢性阻塞性肺疾病急性加重期 | 2016 汇总版 | J44.001 或 J44.101 | 87 |
| 急性单纯性阑尾炎行腹腔镜阑尾切除术 | 2016 年版 | K35.1 / K35.9 | 62 |
| 小儿支气管肺炎 | 2010 年版 | J18.0 | 63 |
| 结肠癌根治切除手术 | 2012 年版 | C18 | 93 |

## 文件

数据位于 `data/disease_samples/`。`shared_vocab.json` 包含 329 个共享词元，`tokenizer.json` 可用 Hugging Face Tokenizers 的 WordLevel 分词器读取。

- 各病种的 `tokens/` 提供映射表、JSON 序列和小端 `uint16` 二进制序列。JSON 同时记录路径阶段、事件和约束掩码。
- `parameters/weights.safetensors` 每病种包含 8 组张量，包括词元嵌入、活跃词元掩码、前缀向量、阶段转移矩阵和 LoRA 矩阵。形状、精度和语义见同目录的 `param_metadata.json`。

## 校验

下载仓库文件后，可运行随附脚本核对词元映射、JSON 与二进制的一致性、张量读取及数值。

```powershell
cd data/disease_samples
uv run --no-project --with numpy --with safetensors python verify_disease_samples.py
```

## 许可

本包的再利用许可证尚未指定。

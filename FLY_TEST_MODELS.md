# FLy 测试 — 完整模型下载列表

## 使用说明

```bash
# 下载单个文件
huggingface-cli download <repo> --include "<filename>" --local-dir ./models/

# 或使用 curl（需要代理）
curl --socks5 127.0.0.1:10808 -L -o models/<filename> \
  "https://huggingface.co/<repo>/resolve/main/<filename>"

# 或用 llama.cpp 内置下载
./build/bin/llama-cli -hf <repo>:<quant> ...
```

---

## 一、Qwen3.5 MTP 系列（最高优先级）

**用途**：draft-mtp + FLy，RS rollback，Mac & 5090 均可

| # | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|---------|---------|------|---------|---------|
| 1 | **Qwen3.5-4B MTP** | `unsloth/Qwen3.5-4B-MTP-GGUF` | Q8_0 | ~4.3 GB | ~6 GB | **Mac / 5090** |
| 2 | **Qwen3.5-9B MTP** | `bartowski/Qwen_Qwen3.5-9B-GGUF` | Q6_K_L | ~7.5 GB | ~10 GB | **Mac / 5090** |
| 3 | **Qwen3.5-27B MTP** | `bartowski/Qwen_Qwen3.5-27B-GGUF` | Q4_K_M | ~18 GB | ~24 GB | **5090 优先**，Mac 64GB |
| 4 | **Qwen3.5-35B-A3B MTP** (MoE) | `unsloth/Qwen3.5-35B-A3B-MTP-GGUF` | Q4_K_M | ~20 GB | ~24 GB | **5090 优先** |
| 5 | **Qwen3.6-27B MTP** | `unsloth/Qwen3.6-27B-MTP-GGUF` | Q4_K_M | ~17 GB | ~22 GB | **5090 优先** |
| 6 | **Qwen3.6-35B-A3B MTP** (MoE) | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | Q4_K_M | ~20 GB | ~24 GB | **5090 优先** |

> **Qwen3.5 vs Qwen3.6**：两者的 MTP + RS rollback 支持一致。Qwen3.6 是新版。测一个即可覆盖。

## 二、Qwen3.5 小模型（快速迭代）

**用途**：快速冒烟测试，K<=W 边界条件

| # | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|---------|---------|------|---------|---------|
| 7 | **Qwen3.5-0.6B MTP** | `unsloth/Qwen3.5-0.6B-MTP-GGUF` | Q8_0 | ~0.7 GB | ~2 GB | **Mac / 5090** |
| 8 | **Qwen3.5-1.7B MTP** | `unsloth/Qwen3.5-1.7B-MTP-GGUF` | Q8_0 | ~1.8 GB | ~3 GB | **Mac / 5090** |

> **0.6B 是最快的测试模型**：加载 < 2s，适合频繁改代码后验证。

## 三、Llama 3.x 系列（draft-simple）

**用途**：draft-simple + FLy，纯 attention 架构（无 RS rollback，走 checkpoint 路径）

| # | 角色 | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|------|---------|---------|------|---------|---------|
| 9 | **Target** | **Llama-3.1-8B Instruct** | `bartowski/Llama-3.1-8B-Instruct-GGUF` | Q6_K_L | ~6.8 GB | ~9 GB | **Mac / 5090** |
| 10 | **Draft** | **Llama-3.2-1B Instruct** | `bartowski/Llama-3.2-1B-Instruct-GGUF` | Q8_0 | ~1.3 GB | ~2 GB | **Mac / 5090** |

> 同系列 pair，预期接受率 60-70%。

| # | 角色 | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|------|---------|---------|------|---------|---------|
| 11 | **Target** | **Llama-3.1-70B Instruct** | `bartowski/Llama-3.1-70B-Instruct-GGUF` | Q4_K_M | ~40 GB | ~48 GB | **5090 32GB ❌** (需 CPU offload) |
| 12 | **Draft** | **Llama-3.2-3B Instruct** | `bartowski/Llama-3.2-3B-Instruct-GGUF` | Q8_0 | ~3.2 GB | ~4 GB | **Mac / 5090** |

> 70B 太大，5090 32GB 装不下。**建议跳过，用 8B 验证即可**。

## 四、Llama 4 系列

**用途**：最新架构测试，纯 attention，checkpoint 路径

| # | 角色 | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|------|---------|---------|------|---------|---------|
| 13 | Target | **Llama-4-Scout-17B** | `bartowski/Llama-4-Scout-17B-Instruct-GGUF` | Q4_K_M | ~10 GB | ~14 GB | **Mac / 5090** |

> Llama 4 无 MTP。作为 draft-simple 的 target model 测试。Draft model 可用 Llama-3.2-1B（跨系列，接受率偏低）。

## 五、Gemma 4 系列（MTP via assistant）

**用途**：Gemma 4 的 MTP 通过独立 assistant model 实现，测试点与 Qwen3.5 不同

| # | 角色 | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|------|---------|---------|------|---------|---------|
| 14 | Target | **Gemma-4-12B-it** | `bartowski/gemma-4-12B-it-GGUF` | Q6_K_L | ~10.5 GB | ~14 GB | **Mac / 5090** |
| 15 | MTP | **Gemma-4-12B-assistant** ⚠️ | 需确认 | Q8_0 | ~2 GB? | ~4 GB | **Mac / 5090** |

> ⚠️ Gemma 4 MTP 需要一个单独的 assistant GGUF 文件。文件名通常包含 "assistant" 字样，需要确认 bartowski 是否提供了 assistant 量化。如果找不到，可以用 draft-simple 方式测试 Gemma 4。

## 六、Mistral 3 系列

**用途**：draft-simple + FLy，与 Llama 不同的 tokenizer/架构

| # | 角色 | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|------|---------|---------|------|---------|---------|
| 16 | Target | **Mistral-Small-3.1-7B** | `bartowski/Mistral-Small-3.1-24B-Instruct-2505-GGUF` 中有小版本 | Q6_K_L | ~6.5 GB | ~9 GB | **Mac / 5090** |

> ⚠️ Mistral 的同系列小 draft model 较少。建议用 Qwen3.5-0.6B 做跨系列 draft（接受率会偏低，但能验证兼容性）。

## 七、Phi-4 系列

**用途**：微软系列，不同 tokenizer 及 chat 模板

| # | 角色 | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|------|---------|---------|------|---------|---------|
| 17 | Target | **Phi-4-mini-instruct** (3.8B) | `bartowski/Phi-4-mini-instruct-GGUF` | Q6_K_L | ~3.5 GB | ~5 GB | **Mac / 5090** |

> 小模型，用于验证跨架构 tokenizer 兼容性。可自投机 (ngram) 或用其他小模型做 draft。

## 八、DeepSeek V3 系列

**用途**：大型 MoE，draft-simple + FLy。MTP 尚未完全集成。

| # | 角色 | 模型 | HF Repo | 推荐量化 | 大小 | 内存需求 | 适合机器 |
|---|------|------|---------|---------|------|---------|---------|
| 18 | Target | **DeepSeek-V3-0324** (685B) | `bartowski/DeepSeek-V3-0324-GGUF` | Q4_K_M | ~350 GB | ❌ | 太大了 |

> ❌ DeepSeek V3 即使 Q4 也 350GB+，Mac 和 5090 都跑不了。**跳过**，除非有 A100/H100 集群。

---

## 汇总：建议下载清单

### 必下（P0 测试覆盖）

```
# 最小测试模型（加载 < 2s，快速迭代）
1. Qwen3.5-0.6B-MTP-Q8_0.gguf           ~0.7 GB   → 快速冒烟
2. Qwen3.5-4B-MTP-Q8_0.gguf              ~4.3 GB   → MTP + RS rollback 主力
3. Llama-3.1-8B-Instruct-Q6_K_L.gguf     ~6.8 GB   → draft-simple target
4. Llama-3.2-1B-Instruct-Q8_0.gguf       ~1.3 GB   → draft-simple draft
                    合计: ~13 GB
```

### 推荐（P1 扩展覆盖）

```
5. Qwen3.5-27B-MTP-Q4_K_M.gguf          ~18 GB    → 大模型 MTP + checkpoint 性能
6. Qwen3.5-35B-A3B-MTP-Q4_K_M.gguf      ~20 GB    → MoE MTP
7. Gemma-4-12B-it-Q6_K_L.gguf           ~10.5 GB  → Gemma 4 架构 + MTP assistant
                    合计: ~48 GB
```

### 可选（P2 宽覆盖）

```
8.  Qwen3.6-27B-MTP-Q4_K_M.gguf          ~17 GB    → 最新 Qwen3.6 对比
9.  Llama-4-Scout-17B-Q4_K_M.gguf        ~10 GB    → Llama 4 架构
10. Mistral-Small-3.1-7B-Q6_K_L.gguf     ~6.5 GB   → Mistral 系列
11. Phi-4-mini-instruct-Q6_K_L.gguf      ~3.5 GB   → Phi 系列
                    合计: ~37 GB
```

### 机器分配

| 机器 | 推荐模型 |
|------|---------|
| **Mac** (Metal) | #1, #2, #3, #4, #7 — 中小模型；#5, #6 如果有 64GB+ 内存 |
| **5090** (32GB CUDA) | #1, #2, #3, #4, #5, #6, #7 — 全部可用（27B Q4 ~18GB 可满载 VRAM） |

---

## 快速下载命令

```bash
MODELS_DIR=~/models

# === P0 必下 ===
# Qwen3.5-0.6B MTP (最小测试模型)
huggingface-cli download unsloth/Qwen3.5-0.6B-MTP-GGUF \
  --include "*.gguf" --local-dir $MODELS_DIR/

# Qwen3.5-4B MTP (主力 MTP 测试)
huggingface-cli download unsloth/Qwen3.5-4B-MTP-GGUF \
  --include "*Q8_0*" --local-dir $MODELS_DIR/

# Llama-3.1-8B (draft-simple target)
huggingface-cli download bartowski/Llama-3.1-8B-Instruct-GGUF \
  --include "*Q6_K_L*" --local-dir $MODELS_DIR/

# Llama-3.2-1B (draft-simple draft)
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  --include "*Q8_0*" --local-dir $MODELS_DIR/

# === P1 推荐 ===
# Qwen3.5-27B MTP (大模型 MTP)
huggingface-cli download bartowski/Qwen_Qwen3.5-27B-GGUF \
  --include "*Q4_K_M*" --local-dir $MODELS_DIR/

# Qwen3.5-35B-A3B MoE MTP
huggingface-cli download unsloth/Qwen3.5-35B-A3B-MTP-GGUF \
  --include "*Q4_K_M*" --local-dir $MODELS_DIR/

# Gemma-4-12B-it
huggingface-cli download bartowski/gemma-4-12B-it-GGUF \
  --include "*Q6_K_L*" --local-dir $MODELS_DIR/
```

> **注意**：`unsloth` 的 repo 命名格式可能是 `unsloth/Qwen3.5-4B-MTP-GGUF` 或 `unsloth/Qwen3.5-4B-GGUF`。如果 MTP 专用的不存在，用普通版本（MTP head 已嵌入 GGUF 中）。bartowski 的 Qwen3.5 系列从 b9180 起也已包含 MTP layers。

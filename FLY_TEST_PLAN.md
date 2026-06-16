# FLy (Training-Free Loosely Speculative Decoding) — 完整测试计划

## 背景

FLy 修改的是 speculative decoding 的**验证阶段**（不改变 draft 生成方式），理论上对以下组合**透明兼容**：

- 所有模型架构（纯 attention / hybrid / MoE）
- 所有 draft 方式（draft-simple / draft-mtp / ngram-*）
- 所有启动方式（llama-speculative / llama-cli / llama-server）

当前已验证：**Qwen3.5 dense + draft-simple + llama-speculative**（Metal 后端，checkpoint 路径）。

## 测试维度总览

| 维度 | 变量 |
|------|------|
| 模型架构 | Qwen3.5 dense, Qwen3.5 MoE, Llama 3.x, Llama 4, Gemma 4, Mistral 3, DeepSeek V3, ... |
| Spec 类型 | draft-simple, draft-mtp, ngram-mod, ngram-simple, ngram-map-k |
| 启动工具 | llama-speculative, llama-cli, llama-server |
| FLy 参数 | threshold (2.0/1.5/4.0), window (6/3/10), K (3/8/25) |
| Temperature | T=0 (greedy), T>0 (stochastic) |
| KV 回滚路径 | RS rollback (Qwen3.5), checkpoint (通用), seq_rm (CUDA 纯 attention) |
| 后端 | Metal (Mac), CUDA (Linux/5090) |

---

## 一、模型矩阵

### A 类：MTP 原生支持（`--spec-type draft-mtp`）

这些模型自带 MTP head，不需要额外 draft model，是 FLy 测试的**最高优先级**。

| # | 模型 | 架构 | RS 回滚 | MTP | 建议量化 | 大小 |
|---|------|------|---------|-----|---------|------|
| A1 | **Qwen3.5-4B** / **Qwen3.6-4B** | qwen35 | ✅ | ✅ | Q8_0 | ~4GB |
| A2 | **Qwen3.5-14B** / **Qwen3.6-14B** | qwen35 | ✅ | ✅ | Q4_K_M | ~8GB |
| A3 | **Qwen3.5-27B** / **Qwen3.6-27B** | qwen35 | ✅ | ✅ | Q4_K_M | ~15GB |
| A4 | **Qwen3.5-35B-A3B** MoE | qwen35moe | ✅ | ✅ | Q4_K_M | ~20GB |
| A5 | **Gemma-4-12B** + assistant | gemma4 + assistant | ❌ | ✅ | Q8_0 | ~13GB |
| A6 | **Step-3.5-Flash** | step35 | ❌ | ✅ | Q4_K_M | ~TBD |

> **A1 是快速迭代测试的最佳选择**：模型小、加载快、支持 RS rollback + MTP + FLy 全特性。

### B 类：draft-simple（需要额外 draft model）

这些需要同系列的 small draft model。

| # | Target 模型 | 架构 | Draft 模型 | 量化 | 大小 |
|---|-----------|------|-----------|------|------|
| B1 | **Llama-3.1-8B** | llama | Llama-3.2-1B | Q8_0 | ~1GB draft + ~8GB tgt |
| B2 | **Llama-4-12B** | llama4 | Llama-3.2-1B* | Q4_K_M | ~8GB |
| B3 | **Mistral-3-7B** (Small) | mistral3 | 同系列 0.5B? | Q4_K_M | ~4GB |
| B4 | **Gemma-2-9B** | gemma2 | Gemma-2-2B | Q8_0 | ~2GB draft + ~9GB tgt |
| B5 | **Phi-3-medium** / **Phi-4** | phi3 | Phi-3-mini | Q4_K_M | ~4GB |
| B6 | **DeepSeek-V3-0324** | deepseek32 | 小模型 (如 Qwen2.5-0.5B) | Q4_K_M | ~ |

> *跨系列 draft model 可用但接受率偏低（40-50% vs 70%+ 同系列）。建议同系列优先。

### C 类：ngram 自投机（任意模型即可，无需 draft model）

| # | 模型 | 目的 |
|---|------|------|
| C1 | **任意 A/B 类模型** | 验证 FLy + ngram-* 组合的兼容性 |

### 建议下载优先级

```
P0（必测）: A1 (Qwen3.5/3.6-4B MTP), B1 (Llama-3.1-8B + Llama-3.2-1B)
P1（重要）: A3 (Qwen3.5-27B MTP), A4 (Qwen3.5 MoE MTP), A5 (Gemma-4 MTP)
P2（扩展）: B3-B5 (Mistral, Gemma2, Phi), A6 (Step3.5), B6 (DeepSeek)
P3（可选）: B2 (Llama-4)
```

---

## 二、测试场景矩阵

### 场景组 1：draft-mtp + FLy（最高优先级）

MTP 是最高效的 draft 方式，FLy + MTP 是论文推荐的组合。

| ID | 模型 | 工具 | T | FLy 参数 | K | 后端 | 关注点 |
|----|------|------|---|---------|---|------|--------|
| 1.1 | A1 (Qwen3.5-4B MTP) | llama-speculative | 0 | 默认(θ=2.0,W=6) | 默认(3) | Metal | 基本正确性，RS rollback |
| 1.2 | A1 (Qwen3.5-4B MTP) | llama-speculative | 0 | 默认 | K=8 | Metal | 中等 K，窗口边界 |
| 1.3 | A1 (Qwen3.5-4B MTP) | llama-speculative | 0.7 | 默认 | 默认 | Metal | T>0 stochastic 路径 |
| 1.4 | A1 (Qwen3.5-4B MTP) | llama-speculative | 0 | θ=1.5,W=6 | K=5 | Metal | 激进阈值 |
| 1.5 | A1 (Qwen3.5-4B MTP) | llama-speculative | 0 | θ=4.0,W=3 | K=5 | Metal | 保守阈值 |
| 1.6 | A1 (Qwen3.5-4B MTP) | llama-speculative | 0 | 默认 | K=25 | Metal | 大 K 稳定性，长窗口 |
| 1.7 | A1 (Qwen3.5-4B MTP) | llama-server | 0 | 默认 | 默认 | Metal | 服务端流式输出 |
| 1.8 | A1 (Qwen3.5-4B MTP) | llama-cli | 0 | 默认 | 默认 | Metal | CLI 交互模式 |
| 1.9 | A1 (Qwen3.5-4B MTP) | llama-speculative | 0 | 默认+debug | 默认 | Metal | Debug trace 输出 |

| ID | 模型 | 工具 | T | FLy 参数 | K | 后端 | 关注点 |
|----|------|------|---|---------|---|------|--------|
| 1.10 | A3 (Qwen3.5-27B MTP) | llama-speculative | 0 | 默认 | K=8 | Metal | 大模型 checkpoint I/O (169MB/save) |
| 1.11 | A3 (Qwen3.5-27B MTP) | llama-speculative | 0 | 默认 | K=8 | CUDA | RS rollback 路径对比性能 |
| 1.12 | A4 (Qwen3.5 MoE MTP) | llama-speculative | 0 | 默认 | 默认 | Metal/CUDA | MoE + RS rollback |
| 1.13 | A4 (Qwen3.5 MoE MTP) | llama-server | 0 | 默认 | K=8 | Meta/CUDA | MoE 服务端 |

| ID | 模型 | 工具 | T | FLy 参数 | K | 后端 | 关注点 |
|----|------|------|---|---------|---|------|--------|
| 1.14 | A5 (Gemma-4-12B+MTP asst) | llama-speculative | 0 | 默认 | 默认 | Metal/CUDA | Gemma 4 MTP assistant 路径 |
| 1.15 | A5 (Gemma-4-12B+MTP asst) | llama-server | 0 | 默认 | 默认 | Metal/CUDA | 服务端 |

### 场景组 2：draft-simple + FLy

验证 FLy 对独立 draft model 场景的兼容性。

| ID | 模型 | 工具 | T | FLy 参数 | K | 后端 | 关注点 |
|----|------|------|---|---------|---|------|--------|
| 2.1 | B1 (Llama3.1-8B + L3.2-1B) | llama-speculative | 0 | 默认 | 默认(3) | Metal | 纯 attention，checkpoint 路径 |
| 2.2 | B1 (Llama3.1-8B + L3.2-1B) | llama-speculative | 0 | θ=1.5,W=8 | K=8 | Metal | 激进参数 |
| 2.3 | B1 (Llama3.1-8B + L3.2-1B) | llama-speculative | 0.7 | 默认 | 默认 | Metal | stochastic 跨系列 |
| 2.4 | B1 (Llama3.1-8B + L3.2-1B) | llama-server | 0 | 默认 | 默认 | Metal | 服务端 |
| 2.5 | B1 (Llama3.1-8B + L3.2-1B) | llama-cli | 0 | 默认 | 默认 | Metal | CLI |

| ID | 模型 | 工具 | T | FLy 参数 | K | 后端 | 关注点 |
|----|------|------|---|---------|---|------|--------|
| 2.6 | B3 (Mistral-3-7B + draft) | llama-speculative | 0 | 默认 | 默认 | Metal | Mistral 系列验证 |
| 2.7 | B4 (Gemma-2-9B + G2-2B) | llama-speculative | 0 | 默认 | 默认 | Metal | Gemma 系列，滑动窗口注意力 |
| 2.8 | B5 (Phi-3/4 + Phi-3-mini) | llama-speculative | 0 | 默认 | 默认 | Metal | Phi 系列验证 |
| 2.9 | B6 (DeepSeek-V3 + draft) | llama-speculative | 0 | 默认 | 默认 | CUDA | DeepSeek MoE 大型模型 |

### 场景组 3：ngram-* + FLy

验证自投机方式与 FLy 的兼容性。

| ID | 模型 | 工具 | Spec 类型 | K | 后端 | 关注点 |
|----|------|------|----------|---|------|--------|
| 3.1 | 任意 (如 Qwen3.5-4B) | llama-speculative | ngram-mod | 64 | Metal | ngram-mod + FLy 组合 |
| 3.2 | 任意 | llama-speculative | ngram-simple | 48 | Metal | ngram-simple + FLy |
| 3.3 | 任意 | llama-speculative | ngram-map-k | 48 | Metal | ngram-map-k + FLy |
| 3.4 | 任意 | llama-server | ngram-mod | 64 | Metal | ngram-mod + FLy + 服务端 |

### 场景组 4：组合 spec types + FLy

验证多个 spec type 链式组合 + FLy。

| ID | 模型 | 工具 | Spec 类型组合 | K | 后端 | 关注点 |
|----|------|------|-------------|---|------|--------|
| 4.1 | A1 (Qwen3.5-4B) | llama-speculative | ngram-mod,draft-mtp | 64 | Metal | ngram 在前，MTP fallback |
| 4.2 | A1 (Qwen3.5-4B) | llama-server | ngram-mod,draft-mtp | 64 | Metal | 多 spec + FLy 服务端 |

### 场景组 5：FLy 边界条件 & 正确性专项

| ID | 场景 | 工具 | 关注点 |
|----|------|------|--------|
| 5.1 | K=2, W=6 (K < W) | llama-speculative | K<=W 边界接受（之前导致死循环） |
| 5.2 | K=3, W=6 (K < W) | llama-speculative | 同上的更典型参数 |
| 5.3 | EOS 出现在 draft 中间 | llama-speculative | 控制 token 保护：应在 EOS 位置 strict reject |
| 5.4 | ChatML 分隔符在 draft 中 | llama-speculative | `<|im_start|>` 等标记不应被 "松散" 接受 |
| 5.5 | 全接受 (100% match) | llama-speculative | FLy 退化为标准 SPD 的行为一致性 |
| 5.6 | 全拒绝 (0% match, high margin) | llama-speculative | 首个 token 即 confident mismatch |
| 5.7 | Deferred accept（单 mismatch, clean window） | llama-speculative | FLy 核心路径 |
| 5.8 | Deferred→reject（mismatch + window 内有 mismatch） | llama-speculative | 级联拒绝正确性 |
| 5.9 | 连续多轮 partial acceptance | llama-speculative | checkpoint restore 循环正确性 |
| 5.10 | 长 prompt (>4096 tokens) | llama-speculative | prompt 很长时的 KV cache 状态 |

### 场景组 6：服务端流式输出正确性

| ID | 场景 | 工具 | 关注点 |
|----|------|------|--------|
| 6.1 | 流式输出 + FLy | llama-server | token 输出是否乱序/重复 |
| 6.2 | 并发请求 + FLy | llama-server | 多 slot 间 spec 状态隔离 |
| 6.3 | 取消 mid-stream + FLy | llama-server | cancel 后资源清理 |
| 6.4 | --spec-fly-debug 日志输出 | llama-server | 服务端 debug trace |

### 场景组 7：性能对比测试

每个性能测试需要跑 **同一 prompt 的三次对比**：（1）无 speculative 的 auto-regressive baseline，（2）标准 SPD（exact-match），（3）FLy SPD。

| ID | 模型 | 工具 | K | 后端 | 对比指标 |
|----|------|------|---|------|---------|
| 7.1 | A1 (Qwen3.5-4B MTP) | llama-speculative | 8 | Metal | t/s, accept%, draft% 对比 |
| 7.2 | A1 (Qwen3.5-4B MTP) | llama-speculative | 8 | CUDA | 同上，CUDA vs Metal |
| 7.3 | A3 (Qwen3.5-27B MTP) | llama-speculative | 8 | Metal | 大模型 checkpoint 开销占比 |
| 7.4 | A3 (Qwen3.5-27B MTP) | llama-speculative | 8 | CUDA | 无 checkpoint 开销的性能上限 |
| 7.5 | B1 (Llama3.1-8B + L3.2-1B) | llama-speculative | 4 | Metal | draft-simple 性能对比 |
| 7.6 | B1 (Llama3.1-8B + L3.2-1B) | llama-speculative | 4 | CUDA | 同上 |

### 场景组 8：回归测试

每次修改 FLy 代码后必须过的快速回归套件。

| ID | 场景 | 工具 | 关注点 |
|----|------|------|--------|
| 8.1 | --spec-fly 关闭 + draft-simple | llama-speculative | FLy 关闭时行为与基线完全一致 |
| 8.2 | --spec-fly 关闭 + draft-mtp | llama-speculative | 同上 |
| 8.3 | --spec-fly 关闭 + ngram-mod | llama-speculative | 同上 |
| 8.4 | --spec-fly 关闭 + server | llama-server | 同上 |
| 8.5 | --spec-fly 未指定（默认关闭） | llama-speculative | 未加参数时的默认行为 |
| 8.6 | spec-fly 但无 spec type | llama-speculative | 错误处理（应该报错还是忽略？） |

---

## 三、已知限制 & 测试注意事项

### 3.1 RS Rollback 限制

- `need_n_rs_seq()` 当前**只为 MTP 启用** RS rollback（`common/common.h:379-385`）
- draft-simple + Qwen3.5 仍走 checkpoint 路径
- 测试 draft-simple 时预期 checkpoint 开销，不计为 bug
- **后续工作**：修好 speculative-simple 的 RS rollback 路径后需重新测试

### 3.2 Checkpoint 开销

- Metal 后端下所有模型（含 Qwen3.5）可能走 checkpoint 路径
- 27B 模型 checkpoint ~169MB，save/restore 可占 50-60% 时间
- 性能对比时需标注使用的是什么回滚路径

### 3.3 MTP 可用模型限制

截至 2026.06，llama.cpp 完整支持 MTP 的模型架构：
- ✅ Qwen3.5 / Qwen3.6（dense + MoE）— 最成熟
- ✅ Gemma 4（via separate assistant model）
- ✅ Step 3.5 Flash
- ❌ DeepSeek V3.2（tensors loaded but MTP not wired in decoder）

### 3.4 FLy 输出缓冲

- `fly_output_buffer` 在 server-context 中定义了但 `push/flushable` 未被调用
- 当前 server 流式输出直接发送 accepted token，无 deferred window 保护
- 这是**已知待实现的功能**，暂不计为 bug，但需要在测试中记录行为

---

## 四、快速验证命令参考

### 基本 FLy + draft-mtp（Qwen3.5-4B）
```bash
# 最快验证：基本功能
./build/bin/llama-speculative \
    -m models/Qwen3.5-4B-Q8_0.gguf \
    --spec-type draft-mtp \
    --spec-fly \
    --spec-draft-n-max 8 \
    -p "Once upon a time" \
    -n 128 --verbose

# 加 debug trace
./build/bin/llama-speculative \
    -m models/Qwen3.5-4B-Q8_0.gguf \
    --spec-type draft-mtp \
    --spec-fly --spec-fly-debug \
    --spec-draft-n-max 8 \
    -p "Hello world" -n 64 --verbose 2>&1 | grep FLY
```

### FLy + draft-simple（Llama 3.1）
```bash
./build/bin/llama-speculative \
    -m models/Llama-3.1-8B-Q4_K_M.gguf \
    -md models/Llama-3.2-1B-Q8_0.gguf \
    --spec-type draft-simple \
    --spec-fly \
    --spec-draft-n-max 5 \
    -p "The capital of France" -n 128 --verbose
```

### FLy + ngram-mod（任意模型，无需 draft model）
```bash
./build/bin/llama-speculative \
    -m models/Qwen3.5-4B-Q8_0.gguf \
    --spec-type ngram-mod \
    --spec-fly \
    --spec-draft-n-max 64 \
    -p "Write a poem about AI" -n 256 --verbose
```

### FLy + MTP + server
```bash
./build/bin/llama-server \
    -m models/Qwen3.5-4B-Q8_0.gguf \
    --spec-type draft-mtp \
    --spec-fly \
    --spec-draft-n-max 8 \
    --host 0.0.0.0 --port 8080
```

---

## 五、用户需确认的内容

1. **目前本地已有的模型**：请告诉我有哪些 GGUF，我更新矩阵避免重复下载
2. **5090 机器 CUDA 是否可用**：是否可以在 192.168.10.2 上做 CUDA 测试
3. **优先测试哪个维度**：P0（最小覆盖）→ P1（重要场景）→ P2（全面覆盖）
4. **时间预算**：需要跑多深（快速冒烟 vs 完整回归 vs 性能对比）

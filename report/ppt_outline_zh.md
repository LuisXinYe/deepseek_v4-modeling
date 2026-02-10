# DeepSeek V4 推理性能分析 — PPT 大纲

---

## 幻灯片 1：封面页

- **标题**：DeepSeek V4 推理性能分析：昇腾 910C vs NVIDIA H20
- 基于 Roofline 模型的性能建模，覆盖四种服务场景
- 硬件平台：昇腾 910C、NVIDIA H20
- 服务场景：8K/4K、32K/4K、128K/4K、256K/4K（输入/输出 token 长度）
- 内容结构：架构、瓶颈、优化、模块深度分析、部署建议
- **可视化建议**：标题文字配合平台 Logo 及场景图标

---

## 幻灯片 2：摘要 — 核心发现

- mHC 内核融合（默认开启）将 Prefill 缩短 3--4 倍；910C 8K 下 mHC 从 84% 降至 36%
- 910C Prefill 吞吐接近持平 — 融合后 8K 差距从 2.57 倍缩小至 1.12 倍
- Decode 在 910C 上受 MoE 权重加载限制（40--56%），在 H20 短上下文下受通信限制（54%）
- V4 的 KV 压缩相比 V3 节省 4.6 倍内存 — 使长上下文实际可服务
- P/D 分离比例：910C 上 1P:1D (8K)、1P:1D (32K)、2P:1D (128K)、3P:1D (256K)
- 910C EP 带宽优势（7.84 倍）支持高 EP MoE 配置，H20 上代价过高
- **可视化建议**：编号高亮卡片或图标网格，总结六项核心发现

---

## 幻灯片 3：DeepSeek V4 架构概览

- 43 层 Transformer：2 层全注意力（ratio=1）、21 层 C4A（4 倍压缩）、20 层 C128A（128 倍压缩）
- MQA：64 个 Q 头、1 个 KV 头、head_dim=512，Q LoRA rank=1,024
- MoE：256 个路由专家（top-6 路由）、1 个共享专家、inter_dim=2,048
- mHC（超级连接）：每个子层均有 FP32 前/后投影 + Sinkhorn 归一化
- Lightning Index：64 个索引头、dim=128、topK=512，用于压缩 KV 条目选择
- KV 压缩：分组投影对 K 和 V 缓存按比例压缩（C4A=4 倍、C128A=128 倍）
- **可视化建议**：架构框图，展示层类型、MoE 路由、mHC 数据流和 KV 压缩机制

---

## 幻灯片 4：V4 vs V3 对比

- 总参数量：V4 ~286B vs V3 ~704B（V4 小 2.5 倍）
- Hidden size：V4 = 4,096 vs V3 = 7,168；层数：V4 = 43 vs V3 = 61
- KV 方案：V4 = MQA + KV 压缩（C4A/C128A） vs V3 = MLA（kv_lora_rank=512）
- 每 token KV 缓存：V4 = 15,168 bytes vs V3 = 70,272 bytes — **节省 4.6 倍**
- V4 新增 mHC（超级连接）和 Lightning Index；V3 均无
- V4 以较小 hidden dim 换取激进压缩 — 使 128K+ 上下文在合理 batch 下可行
- **可视化建议**：并排对比表，用箭头高亮关键架构差异

---

## 幻灯片 5：硬件平台 — 910C vs H20

- Cube 算力（BF16）：910C = 376 TFLOPS，H20 = 148 TFLOPS（910C 2.54 倍优势）
- HBM 带宽：910C = 1,800 GB/s，H20 = 4,000 GB/s（H20 2.22 倍优势）
- EP 带宽：910C = 392 GB/s，H20 = 50 GB/s（910C 7.84 倍优势）
- HBM 容量：910C = 64 GB，H20 = 96 GB（H20 1.5 倍优势）
- Cube:Vec 比：910C = 15.7:1（不均衡），H20 = 3.4:1（更均衡）
- **可视化建议**：并排对比表，用颜色标注各指标优势方（绿色 = 该指标胜出方）

---

## 幻灯片 6：各类别瓶颈总结

- Prefill 为计算受限（CUBE）：注意力投影、MoE 专家矩阵乘法均受 CUBE 限制
- Decode 为访存受限（MEM/COMM）：MoE 权重加载和 KV 缓存读取主导
- mHC：两阶段均为 MEM 受限（融合内核大幅降低绝对时间）
- Lightning Index：COMM 受限（score AllReduce 跨 TP rank）
- 通信：MoE 的 AllToAll dispatch/combine，注意力的 AllReduce，SP 的 AllGather
- **可视化建议**：表格，按类别标注瓶颈类型（CUBE / VEC / MEM / COMM），分 Prefill 和 Decode 颜色编码

---

## 幻灯片 7：Prefill 算子分解 — 910C（融合基线）

- 8K：mHC = 36.3%，Attn Proj = 25.4%，通信 = 18.1%，Attn Compute = 7.8%，Lightning Index = 4.8%
- 32K：mHC = 29.4%，Attn Proj = 20.6%，Attn Compute = 16.2%，通信 = 14.6%，Lightning Index = 13.1%
- 128K：Attn Compute = 31.6%，Lightning Index = 28.2%，mHC = 16.7%，Attn Proj = 11.7%
- 256K：Attn Compute = 39.0%，Lightning Index = 35.5%，mHC = 10.6%，Attn Proj = 7.4%
- 趋势：短上下文 mHC 主导；注意力计算二次增长，128K+ 下成为主要瓶颈
- **可视化建议**：堆叠柱状图，展示 8K/32K/128K/256K 各类别占比

---

## 幻灯片 8：Decode 算子分解 — 910C（融合基线）

- 8K：MoE Routed = 55.9%，Attn Compute = 14.5%，通信 = 12.9%，Lightning Index = 5.7%
- 32K：MoE Routed = 38.1%，Attn Compute = 25.2%，通信 = 14.4%，Lightning Index = 11.3%
- 128K：MoE Routed = 40.3%，Attn Compute = 22.0%，Lightning Index = 16.4%，通信 = 12.4%
- 256K：MoE Routed = 41.1%，Attn Compute = 21.4%，Lightning Index = 16.7%，通信 = 12.2%
- Decode 由 MoE 权重加载主导（38--56%）；长序列下注意力/索引 KV 缓存读取占比增长
- **可视化建议**：堆叠柱状图，展示 8K/32K/128K/256K 各 Decode 类别占比

---

## 幻灯片 9：910C vs H20 瓶颈对比

- Prefill 8K：910C 由 mHC 主导（36.3%） vs H20 由 Attn Proj（36.0%）和通信（26.6%）主导
- 根因：910C 较低 HBM 带宽使 MEM 受限的 mHC 更慢；H20 较低 Cube 算力使瓶颈转向 CUBE 算子
- Decode 8K：910C 由 MoE Routed 主导（55.9%） vs H20 由通信主导（54.2%）
- 根因：910C 较低 HBM 带宽使权重加载占主导；H20 的 50 GB/s EP 使 AllToAll 占主导
- 128K+ 下两个平台均收敛为 MoE 权重加载受限
- **可视化建议**：配对饼图或堆叠柱状图，对比 910C vs H20 Prefill 和 Decode 的瓶颈分布

---

## 幻灯片 10：最优配置 — 8K/4K 和 32K/4K

- 910C 8K：Prefill 1,656 tps/gpu (TP=8,EP=16,DP=2,BS=256)，Decode 181 tps/gpu (TP=4,EP=16,DP=4,BS=512)
- H20 8K：Prefill 1,848 tps/gpu (TP=8,EP=8,DP=1,BS=128)，Decode 252 tps/gpu (TP=8,EP=16,DP=2,BS=512)
- 910C 32K：Prefill 1,340 tps/gpu (TP=8,EP=16,DP=2,BS=64)，Decode 62.2 tps/gpu (TP=4,EP=32,DP=8,BS=512)
- H20 32K：Prefill 1,463 tps/gpu (TP=8,EP=8,DP=1,BS=32)，Decode 137 tps/gpu (TP=4,EP=16,DP=4,BS=256)
- Prefill 吞吐差距：H20 仅高 1.12 倍（8K）、1.09 倍（32K）— 融合后接近持平
- **可视化建议**：分组柱状图，对比 910C vs H20 8K 和 32K 的 Prefill 和 Decode 吞吐

---

## 幻灯片 11：最优配置 — 128K/4K 和 256K/4K

- 910C 128K：Prefill 760 tps/gpu（16 GPUs），Decode 16.7 tps/gpu（32 GPUs）；H20：798 / 47.4 tps/gpu
- 910C 256K：Prefill 482 tps/gpu（16 GPUs），Decode 8.5 tps/gpu（32 GPUs）；H20：497 / 25.6 tps/gpu
- 128K+ 下每 rank 最大 batch 降至个位数，受 KV 缓存内存限制
- H20 Decode 优势扩大：128K 高 2.84 倍，256K 高 3.01 倍（HBM 带宽主导 Decode）
- Prefill 吞吐接近持平：H20 仅高 1.05 倍（128K）、1.03 倍（256K）— 长上下文下 Cube 算力主导
- **可视化建议**：分组柱状图，910C vs H20 128K/256K 各项指标；标注内存约束信息

---

## 幻灯片 12：延迟随场景扩展趋势

- Prefill 延迟：910C 上 330ms -> 1,535ms -> 10,747ms -> 33,923ms（8K -> 32K -> 128K -> 256K）
- 910C 全场景 Prefill 延迟低于 H20 1.68--1.94 倍（Cube 算力优势）
- Decode 延迟极为稳定：910C 上 19.3 -> 19.4 -> 21.0 -> 21.6 ms/step（KV 压缩有效控制开销）
- H20 Decode 延迟：9.0 -> 9.1 -> 9.9 -> 10.1 ms/step — 稳定快约 2 倍（HBM 带宽优势）
- Decode 吞吐急剧下降：910C 上 181 -> 62.2 -> 16.7 -> 8.5 tps/gpu（8K -> 256K）
- **可视化建议**：双轴折线图 — Prefill 延迟和 Decode 吞吐 vs 序列长度，双平台对比

---

## 幻灯片 13：P/D 分离比例

- 公式：N_p/N_d >= (D_tps_instance x input_len) / (P_tps_instance x output_len)
- 910C：1P:1D (8K, 32 GPUs)、1P:1D (32K, 48 GPUs)、2P:1D (128K, 64 GPUs)、3P:1D (256K, 80 GPUs)
- H20：1P:1D (8K, 24 GPUs)、2P:1D (32K, 32 GPUs)、4P:1D (128K, 48 GPUs)、14P:1D (256K, 144 GPUs)
- 256K 下 910C 需 80 GPUs vs H20 需 144 GPUs — 910C 少 44%（因 H20 极端 P:D 比例）
- 内核融合显著改善 910C P:D 比例（128K：原 4:1 降为 2:1）
- **可视化建议**：堆叠柱状图，展示各场景/平台 Prefill GPU 数 + Decode GPU 数；P:D 比例表格

---

## 幻灯片 14：mHC 优化 — 四个级别

- 未融合 FP32：Prefill 10,144ms，mHC = 84.3%（原始基线）
- 融合 FP32（默认）：2,492ms，mHC = 36.0% — 4.07 倍加速，融合消除 HBM 往返
- 融合 FP32 + SP：1,706ms，mHC = 6.6% — 5.95 倍加速，mHC 在 TP rank 间并行化
- 融合 BF16 + SP：1,650ms，mHC = 3.4% — 6.15 倍总加速
- 完全优化后瓶颈迁移至 Attn Proj（38.1%）和通信（28.0%）— CUBE 受限区域
- **可视化建议**：瀑布图，展示各 mHC 优化阶段的时间缩减和每级 mHC 占比

---

## 幻灯片 15：mHC 瓶颈迁移

- 未融合：mHC = 84.3%，Attn Proj = 6.2%，通信 = 4.6% — mHC 压倒一切
- 融合：mHC = 36.0%，Attn Proj = 25.2%，通信 = 18.6% — 分布更均衡
- 融合+SP：mHC = 6.6%，Attn Proj = 36.9%，通信 = 27.1% — 注意力投影成为主导
- 融合 BF16+SP：mHC = 3.4%，Attn Proj = 38.1%，通信 = 28.0% — mHC 已基本消除
- 启示：融合将瓶颈从 MEM（mHC）转移到 CUBE（注意力）— 910C 2.54 倍算力优势的发力点
- **可视化建议**：堆叠柱状图，展示各优化级别的类别占比，箭头标注瓶颈迁移方向

---

## 幻灯片 16：SP/mHC-SP 对比（融合基线）

- 910C 无 SP -> SP+mHC-SP 加速：2.06 倍（8K）、1.79 倍（32K）、1.39 倍（128K）、1.23 倍（256K）
- H20 无 SP -> SP+mHC-SP 加速：2.90 倍（8K）、2.48 倍（32K）、1.78 倍（128K）、1.48 倍（256K）
- 仅 SP：H20 受益更大（8K 下 2.67 倍），因其大幅减少了昂贵的 EP AllToAll（50 GB/s）
- 仅 SP：910C 受益较小（8K 下 1.41 倍），因 EP 带宽本身已较快（392 GB/s）
- 长上下文下加速递减，因二次增长的注意力计算不受 SP/mHC-SP 影响
- **可视化建议**：分组柱状图，展示无 SP / SP / SP+mHC-SP 三种配置在 8K/32K/128K/256K 下的 Prefill 时间

---

## 幻灯片 17：注意力与 KV 缓存分析

- V4 128K 下 KV 缓存：2.18 GB vs V3 MLA：9.21 GB vs 无压缩：11.54 GB — 比 V3 节省 4.2 倍
- 按层分解：C4A 层占 KV 缓存 71--73%，2 层全注意力占 23--25%（尽管仅占 43 层中的 2 层）
- Decode 延迟跨上下文稳定：17.9ms (1K) 至 21.0ms (128K) — 压缩有效控制注意力开销
- 注意力计算扩展：全注意力层 65%->96% 注意力主导（8K->128K）；C128A 维持 31%->54%
- 多比例压缩策略验证：C128A 在超长上下文下仍保持大多数层的高效运行
- **可视化建议**：双图 — KV 缓存大小对比（V4 vs V3 vs 无压缩）和 Decode 延迟 vs 序列长度

---

## 幻灯片 18：按场景部署建议

- 8K/4K（对话/编程）：910C = 32 GPUs (1P:1D)，H20 = 24 GPUs (1P:1D)；两平台均高效
- 32K/4K（文档处理）：910C = 48 GPUs (1P:1D)，H20 = 32 GPUs (2P:1D)；H20 更具成本效益
- 128K/4K（RAG/文档问答）：910C = 64 GPUs (2P:1D)，H20 = 48 GPUs (4P:1D)；P/D 分离必不可少
- 256K/4K（全文档分析）：910C = 80 GPUs (3P:1D)，H20 = 144 GPUs (14P:1D)；910C 更具 GPU 效率
- 通用建议：启用内核融合 + SP（默认），按网络调优 EP，Decode 阶段积极增大 batch
- **可视化建议**：决策流程图或按使用场景分类的推荐配置表，包含平台特定参数和 GPU 数量

---

## 幻灯片 19：行业启示

- 软件优化改变硬件竞争格局：mHC 融合使 910C 从落后 2.57 倍变为接近持平
- KV 压缩是长上下文的必备条件：无压缩时 128K 在合理 batch 下不可行
- HBM 带宽仍是 Decode 的关键差异化因素：对话类负载中 HBM 带宽 > 计算 TFLOPS
- EP 带宽对 MoE 至关重要：910C 7.84 倍优势支持 EP=64；H20 被迫 EP=8
- P/D 分离架构对 128K+ 必不可少（混合服务浪费 60--80% 资源）
- 256K 服务在两平台上代价均高昂（Prefill 延迟 34s/66s）— 需要分块策略
- **可视化建议**：矩阵图，将启示映射到影响程度（高/中/低）和平台相关性

---

## 幻灯片 20：附录 — 方法论与数据来源

- Roofline 模型：每个算子时间 = max(cube, vec, mem) + comm；利用率：计算 50%，HBM 带宽 80%
- 浮点运算：矩阵乘法 [M,K]x[K,N] = MxNxKx2；BF16 = 2 字节，FP32 = 4 字节
- 搜索空间：TP 属于 {1..64}，EP 属于 {1..256}，DP 属于 {1..8}，BS 属于 {1..512}；约束 (TP x DP) % EP == 0
- 通信模型：AllReduce = 2(n-1)/n x vol/BW，AllToAll = (n-1)/n x vol/BW，AllGather = (n-1)/n x vol/BW
- 局限性：通信建模为累加（不与计算重叠），假设 Flash Attention 内存模型
- 原始数据：search_results、pd_ratio_analysis、op_analysis、sp_comparison、mhc_optimization、kv_cache_scaling 等 JSON 文件
- **可视化建议**：Roofline 图，展示算力 vs 访存强度，标注示例算子位置

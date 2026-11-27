# fused_kda_gate_chunk_cumsum：优化目的、实现和生效范围

下面列出当前版本保留的设计，包含交接版本已有思路和本次适配工作。
性能表体现整体版本收益；未做逐项消融的优化，不单独分配加速比。

## 算子做什么

输入 `raw_g[B,T,H,D]` 是模型投影得到的门控原始值，不是历史对话文本。
每个 token、每个 head、每个通道都有自己的 g。`A_log` 每个 head 一个参数，
可展开为 `[H]`；可选 bias 每个 head、每个通道一个，共 `H*D` 个。
真实模型矩阵固定 H=32、D=128，保留全部 38 组采集的 T 和序列边界。

safe gate 为：

`gate[t,h,d] = lower_bound * sigmoid(exp(A_log[h]) * (g[t,h,d] + bias[h,d]))`。

`exp(x)=e^x`；`sigmoid(x)=1/(1+exp(-x))`。当 lower_bound=-5 时，有限输入
得到 (-5,0) 内的门控值。非 safe 模式采用
`-exp(A_log[h]) * softplus(g+bias, beta, threshold)`，并保留原有线性阈值分支。

输出仍为 `[B,T,H,D]`，默认 FP32。对同一序列、同一 head 和通道，在每个
chunk 内沿 token 方向做包含当前位置的前缀和，再乘 `1/ln(2)`。
公式是 `y[t,h,d] = sum(gate[j,h,d], j=chunk_start..t) / ln(2)`。
遇到序列边界或下一个 chunk 重新累计。它为后续 KDA 状态计算准备门控前缀，
自身不执行完整注意力或历史状态更新。

`BT` 是逻辑 chunk 的 token 数，真实矩阵为 64；`BD` 是一个 program
处理的通道数，不是模型 D。一个 program 处理一个 head 的一块 token/channel
数据。各 program 没有跨 chunk 的前缀依赖，因此可以并行调度。

## 1. 用向量 scan 计算前缀，避免构造三角矩阵乘法

**目的：** 前缀和不必把一个 BT×BT 的下三角矩阵与门控矩阵相乘。
原始 Triton 的三角 dot 包含大量乘零/乘一操作；向量 scan 直接表达逐通道前缀。

**实现：** gate 在 FP32 中计算后，执行 `tl.cumsum(b_gate, axis=0)`，再乘
`RCP_LN2`。输入、gate、前缀和及输出在同一个 kernel 内完成，没有额外门控
中间张量的显存往返。原始 Triton 已经融合 gate 和 cumsum；不能把“首次融合”
归为本次新增贡献。scan 思路继承交接版本，本次完成跨平台集成和边界语义补全。

**理论：** 三角 dot 按 BT×BT×BD 的矩阵规模表达问题；scan 只对 BT×BD
数据做前缀组织，其并行依赖深度约 log2(BT)。实际指令数、通信方式由后端编译器
决定。Tensor Core/矩阵单元很强时，较少工作不必然更快，因此 HCU/MetaX
保留短 launch 的矩阵路径，不能宣称 scan 对所有卡、所有 shape 都最优。

**位置：** 通用文件 `kda_gate_cumsum_fwd_kernel`；Ascend 文件
`_ascend_scan_kernel`。MThreads/NVIDIA/PPU 使用通用 scan；Ascend 使用专用 scan。

## 2. HCU/MetaX 在 host 先选择矩阵或 scan，再各自 autotune

**目的：** 小规模 launch 的 scan 通信/调度开销可能超过省下的算术工作；
长输入时矩阵路径的重复运算更突出。一套分块候选不能代替算法选择。

**实现：** `_select_kernel` 用逻辑 head/chunk 组数，与
`ceil(计算单元数 * 每单元最大线程数 / (4 * 参考组 warp 数 * warp_size))`
比较。小于该覆盖量走矩阵，大于等于走 scan。BW1000 对应 400 组，C550
对应 416 组；参考组使用原始候选的最少 2 warp。

**硬件关系：** BW1000 为 80 CU、2560 线程/CU、wave64；C550 为 104 CU、
2048 线程/CU、wave64。这是与机器规模关联的 launch 覆盖估计。
分母中的 4 是校准的 25% 覆盖系数，并非物理极限，也不是实测 occupancy。
最终切换仍需用固定 38 组数据验证，不能把这个启发式说成严格硬件推导。

chunk128 固定 scan，避免原始三角 dot 的资源超限。省略 chunk_indices 的
矩形网格含空闲 program，不能直接用其总数估计有效覆盖；这条路径单独处理。

## 3. HCU/MetaX 相邻 program 优先访问相邻 head

**目的：** 连续 BTHD 布局下，同一 token 的不同 head 数据相邻；一个 head
的远处 token chunk 距离较大。改变 program 与数据的映射，改善局部性。

**实现：** `_program_coordinates` 把二维 program ID 展平后重新解码，
令 head 优先变化。program 数、序列/chunk 边界、结果位置均不变。

**限度：** GPU 不保证严格按 ID 顺序执行，不能说它保证合并所有 program
的访存；这是访问映射的局部性优化。NVIDIA/MThreads/PPU 没有启用该重排。

## 4. 有界 32 位地址索引，必要时保留 64 位

**目的：** 常见模型尺寸不需要所有中间地址都用 64 位，减少宽整数地址计算。

**实现：** HCU/MetaX 的通用路径按包含填充 chunk 的 `(T+BT)*H*D*B`
范围选择 program/chunk 索引位宽。Ascend 专用路径对地址链进行对应选择：
小于 2^31 使用 32 位，达到边界切换 64 位。
Ascend 同时拓宽 dense batch 基址和 cu_seqlens 读取，避免只拓宽最后一步时
上游乘法已经溢出。3 个强制 64 位小张量测试覆盖 dense、indexed、rectangular
分支；没有声称已经实际分配并验证多 GB 极限张量。通用 GPU 极大张量的
完整地址链不在本轮真实 shape 验证范围内。

MThreads 本轮没有启用这个优化，不应搬用之前独立 gate 算子的描述。

## 5. 短输入减少物理行数，逻辑 chunk 大小保持不变

**目的：** T=1/2/4 时，按 64 行计算激活和 scan 会处理大量填充位置。

**实现：** HCU/MetaX 和 Ascend 使用 `ROWS=min(BT,next_power_of_2(T))`。
例如真实 chunk64、T=2 只计算 2 行。逻辑 chunk 的起点仍为 `chunk_id*BT`，
因此不会改变输出的分段含义。MThreads 本轮仍计算完整物理 BT 行。

**正确性：** 原版三角 dot 的 `0*NaN/Inf` 会污染较早位置。scan 额外检查最后
一个非有限 gate；省略物理填充行时，还计算一行代表性 padded gate，保留这类
污染语义。不能简单把填充行全部删掉而忽略原版 NaN 行为。

## 6. HCU 扫描配置与 MetaX 短 safe-gate 布局

HCU scan 单独使用 8 组候选，针对 scan 的寄存器工作量与线程数选配置，
采用单阶段；该 kernel 没有需要多阶段预取的长循环。矩阵路径仍保留原始
6 组候选，不与 scan 混调。

MetaX 的短 safe-gate 在 Gluon 可用时，显式按 `[LANE_T,64/LANE_T]` 分配
wave64 中 token/channel 方向的 lane，按 `[1,num_warps]` 分配 wave。
LANE_T=16/32 让每个通道的 scan 尽量局限在一个 wave 和线程寄存器内，
不同 wave 负责独立通道组。8 组 BD/lane/warp 候选平衡通道连续性、寄存器
工作和并行组数。它只用于短 safe 输入；长输入保留标准 Triton 路径。

已检查的样本编译产物没有共享内存和 CTA barrier；这不是对所有形状的保证。
HCU 的已装 Gluon 编译器缺少 HCUMFMALayout，因此使用标准 Triton，未修改
共享编译器。移除的 evict_first 提示、长输入 Gluon 和无收益候选不计作优化。

## 7. Ascend 将独立 chunk 与通道块暴露给向量核

**基线差异：** 所选 AscendC `.cpp` 在变长路径按“序列×head”分任务，每个
任务循环处理全部 chunk；`ProcessChunk` 又逐 token 循环，执行加载、ApplyGate、
加法、写回及同步。dense 路径已经按 chunk 分任务，不能笼统说所有基线路径
都没有 chunk 并行。

**我们的实现：** 网格为 `(ceil(D/BD),NT,grid_batches*H)`，每个 program
处理独立的 head/channel/chunk。chunk 内用二维向量 gate 和 scan，减少逐 token
控制与同步。变长输入把不同 chunk 也交给调度器，利用 910B4 的 40 个向量核。
program 数增加不等于同时驻留数量，不能把 grid 大小当成硬件并行上限。

**参数复用：** program 内加载一份 A 和 BD 个 bias，广播到行方向。原始 vLLM
Triton 已有这类复用，因此不能称为相对它的新增优化。此次所选 AscendC `.cpp`
在 token 循环中调用 ApplyGate，内部重复读取 A/bias 并计算 exp(A)，与本实现
的 program 内复用存在差异；比较必须明确具体版本和路径。

## 8. Ascend 控制 UB 工作集与完整块访存

**目的：** 二维 scan 有多份活跃 FP32 中间值，过大矩阵或隐式多缓冲容易增加
UB 压力。小 BD 会增加 program 数，较大 BD 减少 program 但增加单块工作集。

**实现与硬件：** 910B4 查询到 UB=196608 字节，即 192 KiB。真实 BT64 时，
BD32/64/128 的单个 FP32 矩阵分别为 `64*BD*4` = 8/16/32 KiB。输入转换、
激活、scan、谓词等中间值也占空间，不能仅凭这一份矩阵推导总峰值或 occupancy。
保留这 3 组候选，由本卡原生 Graph 选参；单 pipeline stage，并在当前算子
launch 参数中传入 `multibuffer=False`。它不修改全局设置，不影响其他算子。
配置中的 `num_warps=4` 是该后端的编译参数，不能套用 NVIDIA 的
“4×32 个线程”来解释 NPU，也没有用它推导向量核的实际驻留率。

当 D 可被 BD 整除，且当前 chunk 的 ROWS 都在序列范围内，使用无 mask 的
加载/写回；尾块和通道尾部保留 mask。目的在于简化完整块谓词和编译生成的
访存控制，不是取消边界保护。

## 9. MThreads 接入与持续 Graph 选参

**接入：** 复用通用 Triton scan，设备/分配/stream 通过仓库 runtime 获取，
使用 MUSA Graph、MUSA Event。原始仓库 libtuner 的显式 CUDA Graph callback
在该镜像调用 CUDA stream 失败，所以给当前算子提供 MUSA callback；未修改
全局 torch.cuda、共享编译器或其他算子。

**硬件联系：** S5000 查询到 60 个计算单元、warp32、每 block 最多 1024
线程。BT64、BD32 为 2048 个元素；2/4/8 warp 对应 64/128/256 线程，粗略为
每线程 32/16/8 元素。更多 warp 可降低每线程 scan 与活跃值压力，但也会增加
资源与通信开销，不能只凭元素/线程比断言最快配置。

**异常与修改：** 首轮固定 38 组中 T63 选到 BD32/2-warp，约 22 µs，
单项 0.889x。直接固定配置五轮测到 2-warp 中位约 21.6 µs、8-warp 约
12.3 µs。MUSA autotune 改为与验收一致的每图 256 次调用，10 次回放预热、
7 个 event 样本，再完整重新选参。全部 6 个候选保留，基线不变。
这是更可靠的设备配置选择；没有额外 Python 启动入口缓存，也不把 host
调度节省算成 Graph 内 kernel 加速。尚未用独立 profiler 证明最初短图排名
变化的底层硬件原因，不能把时钟、缓存等猜测写成已确认结论。

## 10. 编译、接口与测量方面的工程工作

Ascend 的 workload 标量仍加入 LibEntry 和 autotune key，但无用的 TUNE_*
形状元信息改成普通标量参数，减少只因元信息不同产生的 CANN 二进制编译。
影响实际计算的 ROWS、INT64_INDEX 等仍是编译常量。它改善冷编译复用，
不应宣称为已经编译、捕获完成后的 NPU Graph 执行加速。

自动 chunk 索引路径使用设备矩形网格，避免 host 读回构造索引；真实 38 组
保留原采集的预计算索引，没有靠省略基线的这项工作制造加速。
空输入、元数据校验、标准导出、独立输出和 NaN/Inf 兼容是正确性/接口工作。

验收在各自容器内使用设备原生 Graph，每图 256 次调用，5 轮交替顺序配对，
每项取配对加速比的中位数，再对全部 38 项算几何平均。编译/autotune、host
序列边界准备在捕获前完成；捕获前与回放后都校验结果。MThreads 对比冻结的
原版 vLLM Triton 在 MUSA 上编译的版本；Ascend 对比文档指定版本的原生
AscendC 完整融合算子。两者基线不同，不能跨平台横向比较绝对加速比大小。

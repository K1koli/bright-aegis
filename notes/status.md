# 状态

更新时间：2026-10-14 傍晚。**模型服务、隧道、免密、预检、真实 H_0 基线全部就位。唯一剩余阻塞是 Proposer 的 API key。**

## 已完成并验证

| 项 | 结果 |
|---|---|
| HarnessX | 工作分支 `aegis-feat`（= `origin/feat/aegis-experiment`）+ 本地 `bright-graded-fitness`（2 个补丁） |
| 依赖 | `.deps` 手动装 ~75 包（pip 在这台机器上不可用）；`import harnessx` 通过 |
| 数据 | biology 103q/57,359d、economics 103q/50,220d、robotics 101q/61,961d；gold **100%** 命中语料 |
| nDCG@10 | 与官方 `run.py` + trec_eval 公式逐字一致（二值、线性 gain、按 query 宏平均） |
| BM25 自检 | bio **0.2048** vs 已发表 0.189；econ 0.1599 vs 0.149；rob 0.1239 vs 0.136 |
| Tier-2 分级门 | 16 项测试通过；`patches/0001-*.patch` |
| Evolver 模板 | BRIGHT 分支；13 项测试通过；`patches/0002-*.patch` |
| 产物契约 | 离线验证：真实 `run_stage_p` 接受我的 raw schema，counterfactual gate 被**复活**（上游休眠） |
| GAAS | PBS Pro；`gpu_ded` 队列（acl 含 `hpc_wei.lu_group`，上限 1440h）；vLLM 模块 0.19.0 |
| GPU | **NVIDIA H200 / 144GB**，9B 只用 17.66GB，无需量化 |
| 模型 | `Qwen/Qwen3.5-9B`（19GB，非 Base）；`~/models/Qwen3.5-9B` |
| **工具调用** | **`--tool-call-parser qwen3_xml`**（`hermes` 会静默返回空 `tool_calls`）；实测通过 |
| 思考模式 | 默认**关闭**（开：188 token / 1.3s；关：**2 token / 0.0s**），可一键开成进化旋钮 |
| 隧道 | `localhost:8000` → `gaas` 登录节点 → `hpc-gaas-g03:8000`（见 `cluster/tunnel.cmd`） |
| 免密 | 可用（`~/.ssh/id_ed25519`），因此我能全自动操作集群 |
| AEGIS 首跑 | Planner R1 **成功**（landscape 已写）；发现 DeepSeek 400 bug 并修好（补丁 0003） |

## 引擎补丁（`patches/`）

| 补丁 | 内容 |
|---|---|
| 0001 | Tier-2 分级不退化 gate（`gates/fitness.py` + Stage-4 接线），16 项测试 |
| 0002 | Evolver 模板的 `bright` benchmark_context 分支，13 项测试 |
| 0003 | `providers/openai` 丢弃「无 content 且无 tool_calls」的 assistant 消息 |
| 0004 | Bash 在 Windows 上修好 + 每阶段成本上限可调（`aegis/budget.py`） |
| 0005 | Evolver 模板记录 BRIGHT 的真实 config 桶旋钮 |

## AEGIS 首次实跑的成本观测（重要）

```
Planner R1 : exit=done steps=25  cost=$4.576  tokens=1,388,181   landscape 已写
Evolver R1 : exit=done steps=63  cost=$16.316 tokens=4,917,931   candidates=2
Critic  R1 : exit=done steps=27  cost=$6.572  tokens=1,964,159   decision 已写
────────────────────────────────────────────────────────────────
一轮 ≈ $27.5        全部是 DeepSeek API；任务 agent 是自有 GPU，$0
```

**`AegisAgent.budget_per_round_usd` 是死参数**——声明并传递但从未被读取。真正生效的是
每个阶段的 `CostGuardProcessor` 上限，原本硬编码 $100，现在统一走
`AEGIS_MAX_STAGE_COST_USD`（驱动默认设 $6）。

## 首次实跑暴露并已修掉的三个阻塞

1. **Bash 在 Windows 上完全不可用 → tools/processor 两个桶被静默禁用。**
   根因不是沙箱,而是引擎写死了 POSIX 路径 `_WORK_ROOT = "/tmp/harnessx-work"`
   (在 Windows 上变成需要管理员权限的 `C:\tmp\...`),每次调用都在 `mkdir` 就
   抛 WinError 5。修好后又发现第二层:**asyncio 在 Windows 上的 subprocess 走命名
   管道**,而受限环境禁止命名管道 → 改用线程跑阻塞式 `subprocess.run`。
   **实测已可用**(`echo` / `python --version` / `dir` 全部 OK)。
   代价:在本会话里 shell 是 cmd 语义;在普通终端里会自动探测到 Git 的真 bash。

2. **`config` 桶无可调项 → 只能进化 prompt。** 现在 H_0 带一个真实 processor
   `RetrievalPolicyProcessor`,五个 kwargs 都是 config 可编辑项,且
   `Retrieve` 工具会**用策略值限制模型自己要求的 k**;驱动从 applied config 读
   `max_steps`。15 项测试通过;**H_0 基线重测 `0.251712`,与改动前逐位相同**。

3. **成本无刹车** → 见上,`AEGIS_MAX_STAGE_COST_USD`。


## 真实 H_0 基线（本实验的起点数字）

```
biology-dev, 50 条 query, k_rollout=1, τ=0.5
mean nDCG@10 = 0.1766      pass rate = 12% (6/50)
parse rate   = 100%        BM25 单独 = 0.1890
obedience cost = +0.0124   （H_0 只比裸 BM25 低 1.2 分）
2.3 s/query → 50 条 2.0 分钟
```

**H_0 几乎等于裸 BM25**，所以它是与已发表 BM25（18.9）可直接对比的真实基线，不是稻草人。

## 踩过的坑（都已修，记录以免重蹈）

1. **Windows schannel 全坏** → git/curl 必须走 OpenSSL；已写 `.ssh/config` 用 HMAC-SHA2 绕开 UMAC。
2. **pip 不能写自己的临时文件** → 第三方包由 `src/tools/install_wheel.py` 手动解包。
3. **跨 PowerShell→ssh→bash 传多行命令会被引号/CRLF 破坏** → 一律 scp 脚本文件，绝不用参数或 stdin 管道。
4. **PBS 会缓冲运行中作业的 stdout，作业结束才落盘** → 脚本内 `exec > <logfile>`。
5. **PBS 把 `CUDA_VISIBLE_DEVICES` 设成 GPU UUID**，而该 vLLM 只会 `int()` 解析 → 作业脚本里翻译成物理索引（不能简单设 0，同节点有别人）。
6. **`Retrieve` 拿不到语料索引** → `run_round` 漏了绑定；且 ContextVar 跨 asyncio 任务不传递，改用模块级全局兜底。
7. **`max_steps=2` 太紧**（模型会调两次 Retrieve）→ 改 4。
8. **`~/.harnessx` 在工作区外被沙箱拒** → 设 `HARNESSX_HOME`。

## 阻塞

`.env.local` 里 Proposer 三行还是注释状态，需要 DeepSeek-V4-Flash 的 key：

```
BRIGHT_PROPOSER_BASE_URL=https://api.deepseek.com/v1
BRIGHT_PROPOSER_MODEL=deepseek-v4-flash
BRIGHT_PROPOSER_API_KEY=<填这里>
```

填好后跑 `python src/bright_harness/preflight.py`，第 4 步会验证 Proposer 连通，然后就能进 T=3 冒烟。

## 环境备忘

- vLLM 作业：`qsub ~/vllm_serve.pbs`（提交后约 7 分钟可用，大部分时间在 CUDA graph 捕获）
- 查节点：`ssh gaas 'bash ~/probe.sh'`
- 隧道：双击 `cluster/tunnel.cmd`（节点变了要改里面的 `NODE=`）
- 用完还卡：`ssh gaas "qdel <jobid>"`

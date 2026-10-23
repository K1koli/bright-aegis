# HarnessX/AEGIS 跨域泛化实验 — 设计基线

状态：Phase 0 进行中（环境 + 数据 + 适应度地基已落地，等待推理端点与 API key）

---

## 1. 命题与协议

**要验证什么**：AEGIS 那套「不训练模型、只进化 harness 组成」的符号搜索，学到的是**域通用的检索结构**，还是**进化集特化的启发式**。

| 项 | 取值 |
|---|---|
| 进化集 | BRIGHT / Biology（切 dev / heldout） |
| 适应度 | Biology-dev 上的 mean nDCG@10（确定性、零 LLM-judge 成本） |
| 主 OOD | BRIGHT / Economics |
| 第三域 | BRIGHT / Robotics（与 bio/econ 词域与推理风格都远） |
| 反向迁移 | 在 Economics-dev 上进化 → 测 Biology-heldout |
| Task agent | Qwen3.5-9B，远端 OpenAI 兼容端点，**全程冻结** |
| Proposer | DeepSeek-V4-Flash（API） |
| 引擎 | 官方 HarnessX `origin/feat/aegis-experiment`（本地分支 `aegis-feat`） |

协议性质：监督式进化 + held-out 评测（进化集标签可见），不是 TTHE，因此**不构造 label-free 代理信号**，适应度直接用 nDCG@10。

**为什么这个题目成立**：HarnessX 论文 §6.1 明写「所有报告的收益都是在用于演化的同一任务集上测量的；本工作未评估对未见任务的留出泛化」，且其全部验证器都是**二值**任务成功（pass@2）。本实验同时补两个洞：①首次做 held-out 跨域泛化；②首次把 AEGIS 接到**分级**指标上。

---

## 2. 环境事实（实测，非假设）

这台机器不能承载你原设计里的「本地 vLLM 9B」，必须用你的远端端点：

| 事实 | 值 |
|---|---|
| GPU | Quadro P620, 2048 MiB → **无法**本地部署 9B |
| 本地服务 | 仅 11434 (Ollama)；8000/8001/1234 空闲 |
| Python | DSH 自带 3.12；`vllm`/`torch` 均无，且本机不装 |
| git | `C:\Program Files\Git\cmd\git.exe`（不在 PATH） |
| **TLS** | Windows schannel 全挂（`SEC_E_NO_CREDENTIALS`）；curl/git 默认后端失败，**git 必须 `-c http.sslBackend=openssl`** |
| **pip** | 不能写自己的临时文件（EACCES）→ 第三方包一律由 `src/tools/install_wheel.py` 手动解包到 `.deps` |
| **编码** | Python stdout 默认 cp1252，工作区路径含中文 → **所有脚本必须 `PYTHONUTF8=1`** |
| PowerShell | 执行策略禁用 `.ps1` → 不写 shell 包装脚本，改用 `src/_boot.py` 自引导 `sys.path` |

---

## 3. 数据事实（实测）

`src/tools/fetch_bright.py` 下载，`data/bright/raw/{documents,examples}/<domain>-*.parquet`。

| 域 | queries | documents | gold 唯一数 | gold 100% 在语料内 | gold/query (min/med/max) |
|---|---|---|---|---|---|
| biology | 103 | 57,359 | 372 | ✅ | 1 / 3 / 19 |
| economics | 103 | 50,220 | 800 | ✅ | 1 / 3 / 85 |
| robotics | 101 | 61,961 | 520 | ✅ | 1 / 2 / 33 |

官方协议（dataset card + 官方 `run.py` 确认）：`gold_ids` 对应 `documents` 子集；qrels **二值**（每个 gold chunk 记 1，`excluded_ids` 断言非 gold）；指标 **nDCG@10**，经 pytrec_eval/trec_eval 计算，**按 query 宏平均**：

```
nDCG@10(q) = [Σ_{r=1..10} rel_r / log2(r+1)] / [Σ_{r=1..min(R_q,10)} 1 / log2(r+1)]
```

（线性 gain，无 2^rel−1；`R_q` = 该 query 的 gold 数。）这与 `src/bright/metrics.py` 的实现逐字一致。

### 自检：我们的 BM25 vs 已发表 BM25

| 域 | 我们实测 | BRIGHT v4 Table 2 | v1 preprint | 差 |
|---|---|---|---|---|
| biology | **0.2048** | 0.189 | 0.192 | +1.6 |
| economics | **0.1599** | 0.149 | 0.149 | +1.1 |
| robotics | **0.1239** | 0.136 | 0.135 | −1.2 |

差距在 1~2 分带内，来源已知：官方用 gensim `LuceneBM25Model(k1=0.9,b=0.4)` + Lucene analyzer（含词干化/停用词）+ **query 侧 BM25 加权** + top-1000；我们用 `[a-z0-9]+` 分词、无词干化、标准 bag-of-words 打分。**这是实现差异，不是流水线错误**（Anserini 默认 bag-of-words 也会比官方低 1–3 分）。

**头room 参考**：单检索器最佳（v4）biology 30.6（gte-Qwen1.5-7B-instruct）/ economics 22.4、robotics 18.2（Instructor-XL）；公开榜上的 agentic/reranker 系统整体均值已到 60+（跨 12 域均值，与本实验口径不同，不可直接比较）。

> 校正：**22.1 不是摘要数字**，而是 v1 preprint 的最佳**平均值**（gte-Qwen1.5-7B-instruct）；v4 表中最高为 22.5，而 v4 的图表标题仍写 24.3，与该表任何一行都不符——camera-ready 的内部不一致，以表格为准。

切分：`biology-dev` = 50 条，`biology-heldout` = 53 条，由 `(domain, seed)` 的 sha256 决定，跨运行稳定（`src/bright/data.py: make_split`）。

---

## 4. AEGIS 引擎的真实形状（读码结论）

来源：`third_party/HarnessX` @ `aegis-feat`。**与论文措辞有几处必须知道的偏差**：

1. **引擎内部只存在二值信号。** 端到端只有 `passed: bool` / `passed_flags: list[bool]`；`harnessx/aegis/` 内 grep 不到任何 nDCG/reward/分级量。
2. **Stage-4 确定性 gate 完全不看分数。** `run_stage_4` 只拿到 Critic 写的 `decision`、文件路径、refuted signatures。gate 序列为：
   `no_op` → `validate_decision_chain` → **structure**（必填键、`file_changes` 非空、`capability_evidence` 形状、IV-9 bucket/扩展名一致、IV-11 探索响应、IV-12 `iterates_from`、`## Failure Evidence`、≥1 anchor）→ **novelty**（签名未被 refuted）→ 短路跳过（省 LLM 预算）→ **canonicalize**（`HarnessConfig` 可加载可规范化）→ **counterfactual** → **replay smoke** → 全绿才 ship。
3. **论文的「跷跷板约束」在代码里的落点 = counterfactual gate**，且是**逐字相等**的硬比较：把此前 `all(flags)` 为真（即全通过）的任务的已记录事件重放到候选的 processor 上，只要 `task_end` 的 `final_output` 或 `exit_reason` 变了就拒。仓库里 **grep 不到 `seesaw`**。它的盲区很关键：**partial-pass 任务不受保护，且没有任何量级比较**。
4. **带数值阈值的回滚不在引擎里，而在 recipe 驱动层**（GAIA `delta_rate <= -0.05 and delta_count <= -3`）。
5. `regressions.md` 是**事后**产物，且**只喂给 LLM prompt**，从不参与 gate。
6. `stages/adjudicate.py` 是死代码，orchestrator 不引用。
7. **counterfactual gate 在上游是休眠的。** 它只重放顶层 `kind ∈ {after_model, after_tool, task_end}` 的行，而**仓库里没有任何生产代码写这种行**——只有 `tests/aegis/unit/test_gate_counterfactual.py` 有。GAIA 与 tau2 驱动都不写。所以论文里的「跷跷板约束」在**已发布代码里从未触发**，它在两个参考基准上静默返回 ok。（已实测确认：`test_artifacts_offline.py` 第 5 步。）
8. **即使触发，它也保护不了 prompt 候选。** 该 gate 只把**已记录事件**重放通过候选的 processor 链，**不发模型调用**。纯 prompt 改动不会改变 `final_output`（除非候选自带改写输出的 processor），因此必然通过。而 prompt 恰恰是 AEGIS 最高频的编辑桶（论文 §6.2：D2/D4 是最频繁目标）。

这两条把 Tier-2 分级门的必要性从「补量级盲区」升级为「补一个根本没生效的机制」。

**可操作的补救（已实现）**：`kind` 行与 `type` 行可以共存于同一 JSONL——Stage P 只透传未知键，`extract_trace_facts` 忽略未知 `type`。因此驱动层在写 journal 记录之外**追加** `kind` 行，一个文件同时服务两个读者，跷跷板真的会触发。离线验证：`kind rows survived Stage P cleaning` + `gate RAN` 两项均通过。

### 保真度开关（与上游的差异，必须写进论文）

| 开关 | 默认 | 说明 |
|---|---|---|
| `--dormant-seesaw` | **关**（跷跷板**生效**） | 默认按**论文设计**让 counterfactual gate 真的触发。加上这个参数则不写 `kind` 行 → 复刻**已发布代码**的休眠行为 |
| `--fitness-gate` | **关** | 默认只有上游五道门。加上才接入我加的 Tier-2 分级门（patch 0001） |

**已记录的决定（用户指示）**：先按"论文设计"跑（跷跷板生效）。**如果最终跨域泛化结果比预期好，就加 `--dormant-seesaw` 重跑一次**，用来排除"结论是被这道门撑起来的"这一可能——即判断好结果究竟来自方法的泛化能力，还是来自一道上游根本没通电的保险。

**τ=0.5** 是任务逼出来的偏差：BRIGHT 的 nDCG 连续，而引擎的词汇表只有二值。上游基准天然二值，无对应物。这一处必须在论文里交代，并做 τ 敏感性检查。

### 对 BRIGHT 的直接影响

引擎的词汇表是「任务通过/不通过」。BRIGHT 的任务 = 一条 query，产出 = 一个排序，分数 = nDCG@10（连续）。所以**必须做一次方法论选择**（见 §6）。

---

## 5. H_0 设计与编辑空间

BRIGHT 的标准评测是「从全语料检索 top-10」。H_0 采用**智能体形态**（贴合 HarnessX 的前提：harness 组合工具与控制流），而非固定管线：

```
H_0 = QueryUnderstand(直通) → Retrieve(BM25, k=100) → 截断 top-10 → Return(严格 id 列表)
```

- **不做稻草人**：H_0 的检索器就是 BRIGHT 自己报告的稀疏基线（Pyserini BM25, k1=0.9 b=0.4），所以 H_0 ≈ 文献可比的 BM25 数字，一切提升都来自 harness 组成而非换检索器。
- **查询理解存在但平凡**（把 query 原样交给 Retrieve），为进化留出全部空间。
- `max_steps=2`（一次 retrieve + 一次作答）→ H_0 等价于单轮检索。
- **输出契约**：最终消息必须给出可解析的 doc id 排序列表；解析失败 → 空列表 → nDCG=0。这是一个真实的失败模式，正是 prompt/processor 桶该修的。

**四桶编辑空间**（对齐论文的 prompt/tools/config/processor）：

| 桶 | 检索语义下的可编辑项 | 需要新代码？ |
|---|---|---|
| prompt | 查询理解指令、分解指令、输出格式纪律、领域框架 | 否（纯 md/yaml） |
| tools | `Rewrite`、`Decompose`、`HyDE`、`MultiQuery`、`RRF fuse`、`LLMRerank`、`FetchDoc`、`Dedup` | 是（`file://…::Class`） |
| config | `k`、候选池大小、子查询数、融合权重、`max_steps`、温度 | 否 |
| processor | 去重、空结果兜底、格式守卫、预算守卫、多样性过滤 | 是 |

**泄漏防线（硬约束）**：harness 只能看到 query 文本与语料（doc id + content）；`gold_ids`/qrels 只存在于**评测器闭包**内，物理上不可达。Stage-4 增一条确定性检查：候选未访问评测文件。再附污染探针（让 Proposer 闭卷回答若干 Bio/Econ query）。

---

## 6. 【需拍板】分级适应度怎么进 gate

这是唯一会改变「方法本身」的选择，三条路线：

**路线 1 — 二值化（Tier 1，零引擎改动）**
`pass(q) = 1[nDCG@10(q) ≥ τ]`（τ=0.5 先验固定），其余照引擎原样；驱动层记录 mean nDCG 并在聚合回退时 rollback。
- ✅ 零引擎风险；Digester 模板、actionability、regressions、scoreboard hit_rate、counterfactual gate 全部原样可用。
- ❌ 丢掉分级信息：0.31→0.49 完全不可见；AEGIS 实际优化的不是 nDCG。

**路线 2 — Stage-4 分级门（Tier 2，改引擎）**
仿 `set_counterfactual_context` 注入 `set_fitness_context(fitness_fn, probe_qids, incumbent_scores)`，新增 gate `check_fitness_noregress`：在**冻结探针集** P 上跑候选 harness，要求
`mean_nDCG(P|cand) ≥ mean_nDCG(P|incumbent) + margin` 且 `∀q∈P: nDCG(q|cand) ≥ nDCG(q|incumbent) − δ`。
- ✅ 真正忠于论文的「Critic + 确定性 gate」，且是**量级感知**的不退化约束（正好补上 counterfactual gate 的盲区）。
- ❌ 需要改引擎；每个候选多花 |P| 次 harness 执行（|P|=20, K=2, T=3 → 120 次，9B 上可接受）。

**路线 3 — 接受判定移到驱动层（Tier 0）**
引擎不动，驱动层每轮测候选在 dev 上的 nDCG，按 δ 做跷跷板并回滚。
- ✅ 最省事。
- ❌ gate 不再是「发布前」的确定性门；坏 harness 会先上线、下一轮才回滚，与论文架构不符。

**我的建议：路线 1 + 路线 2 并用。** 二值化只为让引擎的既有机制（Digester 的 PASS/FAIL 模式、regressions 分级、counterfactual gate）继续工作；**真正的适应度门用路线 2**，跑在冻结探针集 P ⊂ Biology-dev 上。驱动层再保留路线 3 的聚合回滚作为兜底。
> 这样：`τ` 决定引擎的词汇表，`margin/δ` 决定真正的接受判定，两者都**先验注册**、写死在配置里，不随轮次调整。

---

## 7. 评测协议与统计

- **H\* 只用 Biology-dev 选**，选定即冻结；Economics / Robotics / Biology-heldout 在最终评测前**一次都不跑**。
- 主图：**nDCG@10 vs token 消耗的 Pareto**，而非单点对比——进化出的 harness 必然多花 token，必须排除「只是买了算力」。
- 必要基线：compute-matched H_0（同等 token/调用预算的 best-of-N 或加大 k）、人工设计的强 harness、纯 BM25 / BM25+LLM rerank。
- 统计：per-query 配对 bootstrap CI + Wilcoxon；**≥3 个进化种子**（meta-agent 是随机的，单条轨迹不能作为结论）；报告 final 而非 peak。
- 公平性：Task agent 关键调用 T=0，检索结果 per-query 缓存，使候选比较是配对的。

---

## 8. 效度威胁（提前记录）

1. **Proposer 可能记得 BRIGHT**（公开数据集 + DeepSeek-V4 百万 token 上下文）→ 污染探针 + 禁止 proposer 接触 qrels + 检查候选不读评测文件。
2. **同协议跨域 ≠ 跨任务**：三个域都是 BRIGHT、同指令风格、同评测协议。结论只能声称「域间迁移」，不能声称「任务族间迁移」。
3. **dev 只有 50 条 query**：nDCG 差值噪声大 → 必须给 margin 并报 CI。
4. **9B 的能力下限**：论文在 SWE-bench 上观察到 Qwen3.5-9B 各杠杆命中率崩到 ≈0.05。检索任务上可能出现同样的下限，此时负结果同样有信息量。

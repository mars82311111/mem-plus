# mem-plus v11 — 多业务线 Domain 架构

基于 **mem-plus**（22k⭐ · Benchmark 最高分）融合 SuperMem 增强层。

---

## 核心理念（第一性原则）

**问题本质**：当城问我任何问题时，我需要准确知道他的身份、偏好、原则、进行中的项目。
**核心需求**：**确定性精确召回** + **多业务线隔离**

**架构演进**：
- v5: 单agent精确召回
- v11: **3-tier + Domain Ownership** 多业务线记忆架构

---

## v11 架构（3-tier + Domain Ownership）

```
三层记忆：
  Global Shared (mempalace CLI)
       ↑ promote
  Domain Shared (ChromaDB domain_{name}_shared)
       ↑ promote
  Private (ChromaDB domain_{name}_{agent}_private / agent_{agent}_private)
```

```
用户消息
  ↓
message:preprocessed (hook.ts)
  ↓
mem-plus_cli.py search
  ↓
mempalace 底层检索（Global Shared）
  ↓
+ Domain Shared 检索（--domain 时）
  ↓
+ Agent Private 检索（--agent 时）
  ↓
strip_metadata() → filter_credentials() → boosts → dedup → MMR
```

---

## Collection 命名

| 类型 | Collection 名 | 说明 |
|------|--------------|------|
| Global | `mempalace CLI` | 全公司共享知识 |
| Domain Shared | `domain_{name}_shared` | 某业务线全员可读 |
| Domain Private | `domain_{name}_{agent}_private` | 业务线内某agent私人 |
| Agent Private | `agent_{agent}_private` | 无归属的私人笔记 |

---

## CLI 命令

```bash
# 精确召回（全局）
mem-plus_cli.py search "城的身份 CEO"

# 搜索特定业务线
mem-plus_cli.py search "项目X进展" --domain project_x

# 搜索业务线内私人笔记
mem-plus_cli.py search "草稿想法" --domain project_x --agent planner

# 存储到业务线私人笔记（write-time dedup: 相同内容不会重复存储）
mem-plus_cli.py remember "Planner观察到..." --domain project_x --agent planner

# 晋升知识（Private → Domain Shared）
mem-plus_cli.py promote <memory_id> --from-domain project_x --from-agent planner --to-domain project_x

# 晋升知识（Private → Global Shared）
mem-plus_cli.py promote <memory_id> --from-domain project_x --from-agent planner --to-global

# 查看所有业务线和collection状态
mem-plus_cli.py list-domains --verbose

# 删除记忆
mem-plus_cli.py forget <memory_id> --domain project_x --agent planner
```

> ⚠️ `promote --to-domain <name>` 中的 `<name>` 必须是具体的 domain 名称（如 `strategy`、`product`），不能是字面量 `domain`。

---

## 晋升机制

晋升链：**Private → Domain Shared → Global Shared**

```
子agent发现值得沉淀的经验
        ↓
推送给CEO审核（我）
        ↓
我判断：是否进入Global / 归属Domain / 还是留在private
        ↓
写入对应层级
```

晋升方向：
- `--to global`: 晋升到 Global Shared（所有agent可见）
- `--to {domain}`: 晋升到 Domain Shared（该业务线可见）

---

## 核心文件

| 文件 | 作用 |
|------|------|
| `scripts/mem-plus_cli.py` | v11 召回引擎（19KB），含3-tier搜索 |
| `scripts/mem-plus_reranker.py` | MMR + Levenshtein 去重 |
| `domains/domains.json` | 业务线和Agent配置 |

---

## 数据存储

- **Global Shared**: `~/.mempalace/palace/`（387 drawers）
- **Domain/Private**: `~/.openclaw/memory/chroma/`

---

## v11 新增功能

| 功能 | 来源 | 说明 |
|------|------|------|
| Domain Shared 搜索 | v11 新增 | 业务线全员可读 |
| Domain Private 存储 | v11 新增 | 业务线内私人笔记 |
| promote 晋升 | v11 新增 | Private → Domain → Global |
| list-domains | v11 新增 | 查看所有业务线和collection |
| 注入结果保护 | super-mem | `_injected` 结果不被dedup误删 |

---

## 更新日志

### v11.1 (2026-04-09) — Bug 修复 + bge-m3
- **FIX**: dedup at write time — 相同内容用 content hash 作 ID，完全相同内容不会重复存储
- **FIX**: promote `--to-domain <name>` 语义 — `<name>` 必须是具体 domain 名，不再接受字面量 `domain`
- **FIX**: promote `--from-agent` 逻辑 — 当无 `--from-domain` 时，从 `agent_{agent}_private` 而非 `domain_{domain}_{agent}_private` 读取
- **FIX**: bge-m3 1024-dim 兼容性 — 自动创建 `_v2` collection 处理旧 768-dim 数据；搜索跨 v2 集合同时检索
- **FEAT**: bge-m3 中文 embedding — 替换 nomic-embed-text，中文语义搜索能力大幅提升（水果↔苹果 cos=0.676）
- **FEAT**: 中文同义词扩展 — 类别词自动扩展（水果→苹果/香蕉/橙子...），作为 embedding 的补充保障

### v11 (2026-04-09) — 多业务线 Domain 架构
- **FEAT**: 3-tier + Domain Ownership 多业务线记忆
- **FEAT**: search --domain / --agent 多层搜索
- **FEAT**: remember --domain 写入业务线私人笔记
- **FEAT**: promote 晋升机制（Private → Domain → Global）
- **FEAT**: list-domains 查看所有业务线
- **FIX**: ChromaDB 路径迁移到 ~/.openclaw/memory/chroma
- **FEAT**: dedup 保护 _injected 结果不被误删

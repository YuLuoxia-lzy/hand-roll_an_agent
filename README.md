# Agents0to1

学了一阵 hello_agent（感谢 hello_agent 无私奉献）开始从零落框架。

从零到一的一个基础迷你小框架，用 OpenAI 原生接口，不套 LangChain 或别的脚手架。想法很简单：与其调别人封装好的 `Agent()`，不如自己把「消息怎么拼、工具怎么调、循环在哪一步停」都写一遍——写完再看那些框架，会清楚很多。

后续会添加更多新内容，但是我还没有学，边学边加吧~。

> 本 Readme AI 生成，感谢 deepseek。

## 它做了什么

把一个 LLM 包装成能自己干活的智能体：给它一段人设和一份工具清单，它就能自己判断该不该调工具、调哪个、拿到结果之后接着想，直到给出答案。四种 Agent 的区别，只在于「这个过程怎么组织」。

整个框架的硬依赖只有 `openai` 和 `pydantic`，没有别的框架层封装。

模型不绑定某一家。DeepSeek、OpenAI、通义千问、ModelScope、Kimi、智谱，以及本地跑的 Ollama、vLLM 都能接，换一家通常只要改一个 provider 名字。

## 四种 Agent

| Agent | 干什么 | 什么时候用 |
|---|---|---|
| `SimpleAgent` | 一问一答，记住上下文 | 聊天、问答 |
| `ReActAgent` | 循环调用工具，直到不需要为止 | 要算数、要查资料的任务 |
| `ReflectionAgent` | 写初稿 → 自己评审 → 按意见改，迭代几轮 | 写作、生成代码 |
| `PlanAndSolveAgent` | 先把问题拆成步骤，再一步步做 | 多步推理、数学题 |

`ReActAgent` 的工具调用走的是 OpenAI 的 function calling，省了很多事，也少了很多莫名其妙的解析失败。

## 上手

```bash
git clone git@github.com:YuLuoxia-lzy/hand-roll_an_agent.git
cd hand-roll_an_agent
pip install -e .
```

只想装依赖、不想把框架装进环境的话，`pip install -r requirements.txt` 也行，但那样 `examples/` 里的脚本会 import 不到包。

```bash
cp .env.example .env                       # 把 key 填进去
```

习惯用环境变量的话，`.env` 可以不要：

```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxx        # Linux / macOS
```

然后就能跑了 哈哈

```bash
python examples/simple_chat.py
```

`python -m examples.run_all` 是个菜单，四种 Agent 挑着跑，这条连包都不用装。

## 项目结构

```
hand-roll_an_agent/       # 仓库名带横线，不能直接 import，所以包在里面一层
├── agents0to1/           # 真正被 import 的那个包
│   ├── core/             #   基类与基础设施
│   │   ├── agent.py      #     Agent 基类：拼消息、调 LLM、管历史
│   │   ├── hooks.py      #     扩展点：Hook 五阶段 + 管线 + RunContext
│   │   ├── llm.py        #     多厂商 LLM 客户端
│   │   ├── typedefs.py   #     ToolCall / LLMResponse / AgentEvent / Usage
│   │   ├── config.py     #     配置
│   │   ├── message.py    #     历史消息
│   │   └── exceptions.py #     异常体系
│   ├── classic_agent/    #   四种 Agent 范式
│   ├── hooks/            #   现成的 hook 实现（记忆注入）
│   ├── memory/           #   记忆层：向量库、语义记忆、情景记忆
│   ├── tools/            #   工具系统：注册表、内置工具、工具链、并行执行
│   └── utils/            #   日志等
└── examples/             # 手动跑的示例脚本
```

## 接下来

边学边加，做到哪写到哪：

- [ ] 对话持久化，现在历史只在内存里，进程一退就没了
- [ ] 长期记忆，跨会话记住东西
- [ ] RAG，接一个向量库
- [ ] 多 Agent 协作，现在四种范式都是单打独斗
- [ ] 更多内置工具
- [ ] 其实我更想做一个赛博小镇 这也是学hello agent的一个原因

## 更新日志

### 2026-09-15（下半场：修边界、收旧账）

主要变更：修掉一批**不报错的错** —— 它们不抛异常，只是让答案从某一天开始变差。

- **入库路径不再静默丢数据**：`reindex_file` 改成先算新向量、再删旧数据（原来先删后灌，
  embedding 一挂那份文档就永久查不到）；`index_texts` 的 `texts` 和 `metadatas` 不等长
  现在直接报错，不再被 `zip` 悄悄截断。
- **英文句号 `.` 成为句子边界**，带守卫：纯英文文档原来整篇被当成一句话，而
  `3.14` / `v1.2.3` 不会被切开。
- **新增 `EpisodicHook`**（`agents0to1/hooks/episodic.py`）：`after_run` 落库终于有人实现了，
  之前只有基类的空方法。它只存 `after_run` 递过来的那一轮，天然绕开 `save_agent()`
  的重复入库。
- **工具的失败判定改由执行路径给出**，不再读 `result.startswith("错误")` —— 那个判据
  两头都错：计算器的"计算失败: ..."被判成成功，而正常正文以"错误"开头反被判成失败。
- 补上 `ReflectionAgent` / `PlanAndSolveAgent` 的 `_snapshot_state()`，否则 `load` / `fork`
  出来的 agent 会静默退回默认迭代次数和默认提示词。
- 各文件里注释掉的旧代码（约 420 行）统一移出源文件归档，源文件里只留一行指针。

### 2026-09-15

主要变更：框架化——加一个能力从「改一堆文件」变成「写一个新文件，挂上去」。

- 新增 `core/hooks.py`：五个阶段的扩展点 + `RunContext`。失败策略由 hook 自己声明：`open` 吞掉继续，`closed` 中断本轮。
- 记忆层从基类搬出来，成了 `hooks/memory.py` 一个 hook——改造前散在 9 个文件 178 行里。
- `run()` / `stream_run()` 改为模板方法，子类只写 `_run()`；四种 Agent 都能流式跑了。
- 记忆层留了三处缝：换向量库（`store=`）、换打分公式（`scorer=`）、元数据过滤与删除（`filters=` / `delete`）。
- 修复混库时旧模型的向量参与打分——数值看着正常，排序全错。过滤也改到 SQL 层，不再是取回来再筛。
- `LLMResponse` 新增 `usage` / `model` / `finish_reason`。流式的 usage 藏在最后一个 `choices` 为空的块里，`continue` 一下就会把它跳过。
- 修复 ReAct 流式把工具调用前那段话同时作为 `text` 和 `thinking` 吐两遍。
- 离线测试增加到 8 个文件、全绿，不需要 API key。
- 新增 `examples/one_file_extension.py`（一个文件挂三种能力的验收样例）。

### 2026-09-14

主要变更：

- ReAct 的工具调用由正则解析改为 function calling，工具参数改为结构化字典。
- 所有 LLM 调用统一经 `Agent._chat`，修复 `system_prompt` 未注入的问题。
- 历史改为以 Turn 为单位截断，避免 `tool_calls` 与 `tool` 消息的配对被切断；
- 新增快照 `save` / `load` / `fork`，可从同一份快照分出互不影响的实例。
- `AgentEvent` 新增 `thinking` 类型，用于区分过程叙述与最终答案。
- `core/types.py` 更名为 `core/typedefs.py`。
- provider 检测在多个 key 同时存在时给出告警，并以 `base_url` 判定。
- 重构了文件的结构，之前直接git 会因为名称不一致导致出现问题。

## 协议

MIT

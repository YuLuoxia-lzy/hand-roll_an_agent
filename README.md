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
pip install -r requirements.txt
```

```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxx        # Linux / macOS
```

然后就能跑了 哈哈

## 项目结构

```
Agent_0_to_1/
├── core/                # 基类与基础设施
│   ├── agent.py         #   Agent 基类：拼消息、调 LLM、管历史
│   ├── llm.py           #   多厂商 LLM 客户端
│   ├── types.py         #   ToolCall / LLMResponse / AgentEvent
│   ├── config.py        #   配置
│   ├── message.py       #   历史消息
│   └── exceptions.py    #   异常体系
├── classic_agent/       # 四种 Agent 范式
├── tools/               # 工具系统：注册表、内置工具、工具链、并行执行
├── utils/               # 日志等
└── test/classic_test/   # 手动跑的示例脚本
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

### 2026-09-14

主要变更：

- ReAct 的工具调用由正则解析改为 function calling，工具参数改为结构化字典。
- 所有 LLM 调用统一经 `Agent._chat`，修复 `system_prompt` 未注入的问题。
- 历史改为以 Turn 为单位截断，避免 `tool_calls` 与 `tool` 消息的配对被切断；
- 新增快照 `save` / `load` / `fork`，可从同一份快照分出互不影响的实例。
- `AgentEvent` 新增 `thinking` 类型，用于区分过程叙述与最终答案。
- `core/types.py` 更名为 `core/typedefs.py`。
- provider 检测在多个 key 同时存在时给出告警，并以 `base_url` 判定。

## 协议

MIT

"""Plan and Solve Agent实现 - 分解规划与逐步执行的智能体"""

from typing import Optional, List, Dict, Any
from ..core.agent import Agent
from ..core.llm import Agents0to1
from ..core.config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


SUBMIT_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "把任务拆解成按顺序排列的执行步骤,并提交",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "items": {"type": "string", "description": "一个可独立执行的子任务"},
                    "description": "按执行顺序排列的子任务列表"
                }
            },
            "required": ["plan"]
        }
    }
}


# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

class Planner:
    """规划器 - 负责将复杂问题分解为简单步骤"""

    def __init__(self, agent: Agent, prompt_template: Optional[str] = None):
        if not isinstance(agent, Agent):
            raise TypeError(
                f"Planner 现在接收 Agent 实例, 不再是 LLM 客户端; 收到的是 {type(agent).__name__}。"
            )
        self.agent = agent
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划

        Args:
            question: 要解决的问题
            **kwargs: LLM调用参数

        Returns:
            步骤列表

        注意: 规划 不带对话历史。计划是针对"这一个问题"的一次性推理,
        system 仍由 agent._chat 注入, 所以人设照常生效。
        """
        prompt = self.prompt_template.format(question=question)

        # 用 tools= 关键字传: 上一版是位置参数 invoke(messages, [SUBMIT_PLAN_TOOL]),
        # 一旦函数签名调整(比如加个 temperature), 工具清单就会被绑到别的参数上, 且不报错。
        response = self.agent._chat(
            [{"role": "user", "content": prompt}],
            tools=[SUBMIT_PLAN_TOOL],
            **kwargs,
        )
        if not response.has_tool_calls:
            return []

        plan = response.tool_calls[0].arguments.get("plan")
        if not isinstance(plan, list):
            return []

        return plan

        # try:
        #     # 提取Python代码块中的列表
        #     plan_str = response_text.split("```python")[1].split("```")[0].strip()
        #     plan = ast.literal_eval(plan_str) #将字符串转为python可识别的 在这里为转为list
        #     return plan if isinstance(plan, list) else []
        # except (ValueError, SyntaxError, IndexError) as e:
        #     print(f"❌ 解析计划时出错: {e}")
        #     print(f"原始响应: {response_text}")
        #     return []
        # except Exception as e:
        #     print(f"❌ 解析计划时发生未知错误: {e}")
        #     return []

class Executor:
    """执行器 - 负责按计划逐步执行"""

    def __init__(self, agent: Agent, prompt_template: Optional[str] = None):
        if not isinstance(agent, Agent):
            raise TypeError(
                f"Executor 现在接收 Agent 实例, 不再是 LLM 客户端; 收到的是 {type(agent).__name__}。"
            )
        self.agent = agent
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按计划执行任务

        Args:
            question: 原始问题
            plan: 执行计划
            **kwargs: LLM调用参数

        Returns:
            最终答案
        """
        history = ""
        final_answer = ""

        for i, step in enumerate(plan, 1):
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )
            # 每一步同样是"任务内部的分步提示", 不带对话历史 ——
            response_text = self.agent._chat(
                [{"role": "user", "content": prompt}], **kwargs
            ).content or ""

            logger.info("步骤 %d/%d 完成: %s", i, len(plan), step)
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text

        return final_answer

class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent - 分解规划与逐步执行的智能体
    
    这个Agent能够：
    1. 将复杂问题分解为简单步骤
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案
    
    特别适合多步骤推理、数学问题、复杂分析等任务。
    """
    
    def __init__(
        self,
        name: str,
        llm: Agents0to1,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化PlanAndSolveAgent
        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            custom_prompts: 自定义提示词模板 {"planner": "", "executor": ""}
        """
        super().__init__(name, llm, system_prompt, config)

        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        # 传 self(Agent) 而不是 self.llm —— 规划器和执行器都要通过 Agent 调 LLM,
        # system_prompt 才真的会进请求体。
        self.planner = Planner(self, planner_prompt)
        self.executor = Executor(self, executor_prompt)
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Plan and Solve Agent
        
        Args:
            input_text: 要解决的问题
            **kwargs: 其他参数
            
        Returns:
            最终答案
        """
        # 规划器自己负责带 system(_chat 注入), 这里不再拼消息传进去
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            self._record_turn(input_text, final_answer)
            return final_answer

        final_answer = self.executor.execute(input_text, plan, **kwargs)

        #  保存到历史记录 —— 注意记的是"原始问题 -> 最终答案",
        #  中间的 plan 和每一步结果属于任务内部的工作记忆, 不进对话历史。
        self._record_turn(input_text, final_answer or "")

        return final_answer


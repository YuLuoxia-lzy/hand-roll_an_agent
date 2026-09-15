"""Reflection Agent实现 - 自我反思与迭代优化的智能体"""

from typing import Optional, List, Dict, Any
from ..core import Agent, Agents0to1, Config
from ..utils.logging import get_logger

logger = get_logger(__name__)

# 默认提示词模板
DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

class Scratchpad:
    """
    
    """
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """向记忆中添加一条新记录"""
        self.records.append({"type": record_type, "content": content})

    def get_trajectory(self) -> str:
        """将所有记忆记录格式化为一个连贯的字符串文本"""
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 (代码) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip() #去除两侧空白字符

    def get_last_execution(self) -> str:
        """获取最近一次的执行结果"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""    





class ReflectionAgent(Agent):
    """
    Reflection Agent - 自我反思与迭代优化的智能体  历史记录都没有被使用到

    这个Agent能够：
    1. 执行初始任务
    2. 对结果进行自我反思
    3. 根据反思结果进行优化
    4. 迭代改进直到满意

    特别适合代码生成、文档写作、分析报告等需要迭代优化的任务。

    支持多种专业领域的提示词模板，用户可以自定义或使用内置模板。
    """
    def __init__(
        self,
        name: str,
        llm: Agents0to1,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompt: Optional[Dict[str, str]] = None,
        hooks: Optional[list] = None,
        agent_id: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt, config, hooks=hooks, agent_id=agent_id)
        self.max_iterations = max_iterations
        self.scratch = Scratchpad()
        self.custom_prompt = custom_prompt if custom_prompt else DEFAULT_PROMPTS

    def _run(self, input_text: str, **kwargs) -> str:
        """
        运行Reflection Agent

        Args:
            input_text: 任务描述
            **kwargs: 其他参数

        Returns:
            最终优化后的结果(str —— 与基类 Agent.run 的签名保持一致)
        """
        self.scratch = Scratchpad() #本轮草稿轨迹, 每次 run 重来

        # 【记忆只喂给**初稿**, 不喂给评审和改写】—— 这条策略没变, 变的是谁来执行
        #
        # 为什么只喂初稿: 后面几次调用的最后一条消息是"模型自己的草稿/评审意见",
        # 不是用户问题。拿它去检索, 检索出来的东西和被检索的内容毫无关系;
        # 而且记忆混进评审意见里会变成"记忆在评自己", 越评越偏。
        #
        # 改造前: 这里手抄三行 self._memory_context(input_text), 然后把结果拼进
        # initial_prompt —— 因为这个 Agent 直接用 _chat, 不走 _build_messages,
        # 基类的注入对它不生效。
        # 改造后: MemoryHook 默认 once_per_turn=True, "一轮里只喂第一次 LLM 调用",
        # 而本 Agent 这一轮的第一次调用**就是初稿** —— 策略一模一样,
        # 但执行它的是 hook, 不是这里的一段手抄代码。
        #
        # 想改成"评审也带记忆"? 换个 once_per_turn=False 的 hook 就行, 不用动这个文件。
        initial_prompt = self.custom_prompt["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.scratch.add_record("execution", initial_result)

        for i in range(self.max_iterations):
            last_result = self.scratch.get_last_execution()
            reflect_prompt = self.custom_prompt["reflect"].format(
                task = input_text,
                content = last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.scratch.add_record("reflection", feedback)

            logger.info("第 %d 轮评审反馈: %s", i + 1, feedback)

            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                logger.info("评审认为无需改进, 提前结束迭代。")
                break

            refine_prompt = self.custom_prompt["refine"].format(
                task = input_text,
                last_attempt = last_result,
                feedback = feedback
            )

            refine_result = self._get_llm_response(refine_prompt, **kwargs)
            self.scratch.add_record("execution", refine_result)

        final_result = self.scratch.get_last_execution()

        self._record_turn(input_text, final_result or "")

        return final_result
        


        #摒弃了长期记忆 如何实现长期记忆与短期记忆并行 且不冲突？
    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """
        调用LLM并获取完整响应。
        """
        return self._chat([{"role": "user", "content": prompt}], **kwargs).content or ""
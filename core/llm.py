"""大模型接口与创建 统一化！"""

import os
import logging
from typing import Literal, Optional, Iterator
from openai import OpenAI

from .exceptions import *
from .types import LLMResponse, ToolCall, parse_tool_calls, safe_parse_arguments
from .config import Config

# 各模块自己 getLogger(__name__)
logger = logging.getLogger(__name__)

API_PROVIDERS = Literal["openai", "deepseek", "qwen", "vllm"]

class Agents0to1:
    """
    为后续agent开发定制的大模型客户端
    用于调用各类AI厂家API 待补充 默认使用 流式相应
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[API_PROVIDERS] = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        支持自动检测provider或使用统一的LLM_*环境变量配置。

        Args:
            model: 模型名称，如果未提供则从环境变量LLM_MODEL_ID读取
            api_key: API密钥，如果未提供则从环境变量读取
            base_url: 服务地址，如果未提供则从环境变量LLM_BASE_URL读取
            provider: LLM提供商，如果未提供则自动检测
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间，从环境变量LLM_TIMEOUT读取，默认60秒
            config: 默认参数配置
        """
        self.config = config or Config() 
        self.model = model or self.config.default_model or os.getenv("LLM_MODEL_ID") 
        self.temperature = temperature if temperature is not None else self.config.temperature # 必须 is not None, 0 是合法值, or 会把它吞掉
        self.max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        self.last_response: Optional[LLMResponse] = None

        # 自动检测provider或使用指定的provider

        self.provider = (provider or self.config.default_provider
                 or self._auto_detect_provider(api_key, base_url))


        # 根据provider确定API密钥和base_url
        key, url = self._resolve_credentials(api_key, base_url)

        self.api_key = api_key or key
        self.base_url = base_url or url

        # 验证必要参数 放在后面是为了确定确实没救了
        if not self.model:
            self.model = self._get_default_model()
        if not all([self.api_key, self.base_url]):
            logger.error("API密钥和服务地址必须被提供或在环境变量中定义。")
            raise ConfigException("API密钥和服务地址必须被提供或在环境变量中定义。")

        # 创建OpenAI客户端
        self._client = self._create_client()

    """
    以下4个私有函数 直接cv自hello agent 哈哈 懒了... 用于构建 API链接
    包括寻找provider
    根据provider查找key 与 url
    以上两步都没用 就用默认模型
    以及创建链接
    """

    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """
        自动检测LLM提供商

        检测逻辑：
        1. 优先检查特定提供商的环境变量
        2. 根据API密钥格式判断
        3. 根据base_url判断
        4. 默认返回通用配置
        """
        # 1. 检查特定提供商的环境变量
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        # 2. 根据API密钥格式判断
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        if actual_api_key:
            actual_key_lower = actual_api_key.lower()
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            elif actual_key_lower == "ollama":
                return "ollama"
            elif actual_key_lower == "vllm":
                return "vllm"
            elif actual_key_lower == "local":
                return "local"
            elif actual_api_key.startswith("sk-") and len(actual_api_key) > 50:
                # 可能是OpenAI、DeepSeek或Kimi，需要进一步判断
                pass
            elif actual_api_key.endswith(".") or "." in actual_api_key[-20:]:
                # 智谱AI的API密钥格式通常包含点号
                return "zhipu"

        # 3. 根据base_url判断
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api.openai.com" in base_url_lower:
                return "openai"
            elif "api.deepseek.com" in base_url_lower:
                return "deepseek"
            elif "dashscope.aliyuncs.com" in base_url_lower:
                return "qwen"
            elif "api-inference.modelscope.cn" in base_url_lower:
                return "modelscope"
            elif "api.moonshot.cn" in base_url_lower:
                return "kimi"
            elif "open.bigmodel.cn" in base_url_lower:
                return "zhipu"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                # 本地部署检测 - 优先检查特定服务
                if ":11434" in base_url_lower or "ollama" in base_url_lower:
                    return "ollama"
                elif ":8000" in base_url_lower and "vllm" in base_url_lower:
                    return "vllm"
                elif ":8080" in base_url_lower or ":7860" in base_url_lower:
                    return "local"
                else:
                    # 根据API密钥进一步判断
                    if actual_api_key and actual_api_key.lower() == "ollama":
                        return "ollama"
                    elif actual_api_key and actual_api_key.lower() == "vllm":
                        return "vllm"
                    else:
                        return "local"
            elif any(port in base_url_lower for port in [":8080", ":7860", ":5000"]):
                # 常见的本地部署端口
                return "local"

        # 4. 默认返回auto，使用通用配置
        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """根据provider解析API密钥和base_url"""
        if self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
            return resolved_api_key, resolved_base_url

        elif self.provider == "qwen":
            resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolved_api_key, resolved_base_url

        elif self.provider == "kimi":
            resolved_api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.moonshot.cn/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "zhipu":
            resolved_api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
            return resolved_api_key, resolved_base_url

        elif self.provider == "ollama":
            resolved_api_key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
            resolved_base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "vllm":
            resolved_api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "vllm"
            resolved_base_url = base_url or os.getenv("VLLM_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "local":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or "local"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        else:
            # auto或其他情况：使用通用配置，支持任何OpenAI兼容的服务
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            return resolved_api_key, resolved_base_url

    def _create_client(self) -> OpenAI:
        """创建OpenAI客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def _get_default_model(self) -> str:
        """获取默认模型"""
        if self.provider == "openai":
            return "gpt-3.5-turbo"
        elif self.provider == "deepseek":
            return "deepseek-chat"
        elif self.provider == "qwen":
            return "qwen-plus"
        elif self.provider == "modelscope":
            return "Qwen/Qwen2.5-72B-Instruct"
        elif self.provider == "kimi":
            return "moonshot-v1-8k"
        elif self.provider == "zhipu":
            return "glm-4"
        elif self.provider == "ollama":
            return "llama3.2"  # Ollama常用模型
        elif self.provider == "vllm":
            return "meta-llama/Llama-2-7b-chat-hf"  # vLLM常用模型
        elif self.provider == "local":
            return "local-model"  # 本地模型占位符
        else:
            # auto或其他情况：根据base_url智能推断默认模型
            base_url = os.getenv("LLM_BASE_URL", "")
            base_url_lower = base_url.lower()
            if "modelscope" in base_url_lower:
                return "Qwen/Qwen2.5-72B-Instruct"
            elif "deepseek" in base_url_lower:
                return "deepseek-chat"
            elif "dashscope" in base_url_lower:
                return "qwen-plus"
            elif "moonshot" in base_url_lower:
                return "moonshot-v1-8k"
            elif "bigmodel" in base_url_lower:
                return "glm-4"
            elif "ollama" in base_url_lower or ":11434" in base_url_lower:
                return "llama3.2"
            elif ":8000" in base_url_lower or "vllm" in base_url_lower:
                return "meta-llama/Llama-2-7b-chat-hf"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                return "local-model"
            else:
                return "gpt-3.5-turbo"

    def _build_request_params(self, messages: list[dict], tools=None, stream: bool = False, **kwargs) -> dict:
        """
        组装请求参数 —— invoke 和 stream_invoke 共用这一处, 保证两条路的参数规则一致。

        实测: 把 tools=None / max_tokens=None 直接传给 create(),
        请求体里就真的出现 "tools": null —— 部分 OpenAI 兼容服务(vLLM/网关)会因此返回 400。
        所以"没有值"的参数要整个不放进字典, 而不是放一个 None 进去。
        """
        params: dict = {
            "model": self.model,
            "messages": messages,
        }

        # temperature: 调用方没传(或传了 None)就回落到实例配置; 仍然是 None 就干脆不发这个字段
        temperature = kwargs.pop("temperature", None)
        if temperature is None:
            temperature = self.temperature
        if temperature is not None:
            params["temperature"] = temperature

        # max_tokens 同理。Config.max_tokens 默认就是 None,
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = self.max_tokens
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # 空列表和 None 一样表示"本轮没有工具", 都不该出现在请求体里
        if tools:
            params["tools"] = tools

        if stream:
            params["stream"] = True

        # 其余参数(seed、tool_choice、response_format 等)原样透传, 同样过滤掉 None
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return params

    def invoke(self, messages: list[dict[str, str]], tools = None, **kwargs) -> LLMResponse:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        try:
            params = self._build_request_params(messages, tools=tools, **kwargs)
            response = self._client.chat.completions.create(**params)  #调用api 配置参数 得到回答
            message = response.choices[0].message
            return LLMResponse(
                content=message.content,
                tool_calls=parse_tool_calls(message.tool_calls)
            ) #输出模型回答的 第一个内容 一般只有一个
        except Exception as e:
            logger.exception("LLM调用失败: %s", e)
            # from e 保留原始异常链 —— 否则排查时只能看到 LLMException, 看不到底下
            # 到底是连接超时还是 401, 把真正的原因吃掉。
            raise LLMException(f"LLM调用失败:{str(e)}") from e

    def stream_invoke(self, messages: list[dict[str, str]], tools = None, temperature: Optional[float] = None, **kwargs) -> Iterator[str]:
        """
        流式调用LLM。适用于长任务。

        last_response 的语义:
            流被【完整消费】 -> 写入本次的完整响应(含 tool_calls)
            消费者提前 break  -> 停在 yield 处, 赋值语句永远不执行, 保持旧值
        所以每次进来先把它清成 None: 清掉旧值, 调用方就不会拿到"上一轮的响应"
        """
        self.last_response = None

        try:
            content_parts = []
            tool_parts: dict = {}
            # 走和 invoke 同一套参数组装逻辑, 唯一区别是 stream=True
            params = self._build_request_params(
                messages, tools=tools, stream=True, temperature=temperature, **kwargs
            )
            response = self._client.chat.completions.create(**params)

            last_key = None
            for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    yield delta.content

                for tc in delta.tool_calls or []:     # 工具调用: 分片到达, 按index拼接
                    if tc.index is not None:
                        key = tc.index
                    elif tc.id:
                        key = tc.id
                    elif last_key is not None:
                        key = last_key
                    else:
                        key = len(tool_parts)
                    last_key = key

                    part = tool_parts.setdefault(key, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        part["id"] = tc.id
                    if tc.function and tc.function.name:
                        part["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        part["arguments"] += tc.function.arguments

            # 流完整结束了, 才把完整响应挂到实例上
            self.last_response = LLMResponse(
                content="".join(content_parts) or None,
                tool_calls=[
                    ToolCall(
                        id=p["id"],
                        name=p["name"],
                        arguments=self._parse_stream_arguments(p["arguments"], p["name"]),
                    )
                    for p in tool_parts.values()
                ]
            )

        except Exception as e:
            logger.exception("调用LLM API时发生错误: %s", e)
            raise LLMException(f"LLM调用失败: {str(e)}") from e

    @staticmethod
    def _parse_stream_arguments(raw: str, tool_name: str) -> dict:
        """解析流式拼出来的工具参数, 坏 JSON 记一条警告并退回空字典"""
        if not raw:
            return {}

        parsed = safe_parse_arguments(raw)
        if not parsed:
            logger.warning(
                "工具 '%s' 的参数不是合法 JSON(可能是流被截断), 已按空参数处理。原始内容: %.200s",
                tool_name, raw,
            )
        return parsed


    # def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
    #     """
    #     非流式调用LLM，返回完整响应。
    #     适用于不需要流式输出的场景。
    #     """
    #     try:
    #         response = self._client.chat.completions.create(
    #             model=self.model,
    #             messages=messages,
    #             temperature=kwargs.get('temperature', self.temperature),
    #             max_tokens=kwargs.get('max_tokens', self.max_tokens),
    #             **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
    #         )  #调用api 配置参数 得到回答
    #         return response.choices[0].message.content #输出模型回答的 第一个内容 一般只有一个
    #     except Exception as e:
    #         logging.exception(f"LLM调用失败:{str(e)}")
    #         raise LLMException(f"LLM调用失败:{str(e)}")        

    # def stream_invoke(self, messages: list[dict[str, str]], temperature: Optional[float] = None, **kwargs) -> Iterator[str]:
    #     """
    #     流式调用LLM。
    #     适用于长任务。
    #     """        
    #     try:
    #         response = self._client.chat.completions.create(
    #             model=self.model,
    #             messages=messages,
    #             temperature=temperature if temperature is not None else self.temperature,
    #             max_tokens=kwargs.get('max_tokens', self.max_tokens),
    #             stream=True,
    #             **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
    #         )

    #         for chunk in response:
    #             content = chunk.choices[0].delta.content or ""
    #             if content:
    #                 logging.info(content)
    #                 yield content
    #     except Exception as e:
    #         logging.exception(f"❌ 调用LLM API时发生错误: {e}")
    #         raise LLMException(f"LLM调用失败: {str(e)}")
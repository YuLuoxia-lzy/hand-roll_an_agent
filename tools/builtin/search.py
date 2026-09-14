"""搜索工具 - 原生搜索实现"""

import os
from typing import Optional, Dict, Any, List

from ..base import Tool, ToolParameter
from ...core.exceptions import Agents0to1Exception
from ...utils.logging import get_logger

logger = get_logger(__name__)

class SearchTool(Tool):
    """
    智能混合搜索工具
    支持多种搜索引擎后端，智能选择最佳搜索源：
    1. 混合模式 (hybrid) - 智能选择TAVILY或SERPAPI
    2. Tavily API (tavily) - 专业AI搜索
    3. SerpApi (serpapi) - 传统Google搜索
    """

    def __init__(self, backend: str = "hybrid", tavily_key: Optional[str] = None, serpapi_key: Optional[str] = None):
        super().__init__(
            name="search",
            description="一个智能网页搜索引擎。支持混合搜索模式，自动选择最佳搜索源。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
        )
        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.available_backends = []
        self._setup_backends()

    def _setup_backends(self):
        """设置搜索后端"""
        # 检查Tavily可用性
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
                logger.info("Tavily搜索引擎已初始化")
            except ImportError:
                logger.warning("Tavily未安装，无法使用Tavily搜索")
        else:
            logger.warning("TAVILY_API_KEY未设置")

        # 检查SerpApi可用性
        if self.serpapi_key:
            try:
                import serpapi
                self.available_backends.append("serpapi")
                logger.info("SerpApi搜索引擎已初始化")
            except ImportError:
                logger.warning("SerpApi未安装，无法使用SerpApi搜索")
        else:
            logger.warning("SERPAPI_API_KEY未设置")

        # 确定最终使用的后端
        if self.backend == "hybrid":
            if self.available_backends:
                logger.info("混合搜索模式已启用，可用后端: %s", ', '.join(self.available_backends))
            else:
                logger.warning("没有可用的搜索后端，请配置API密钥")
        elif self.backend == "tavily" and "tavily" not in self.available_backends:
            logger.warning("Tavily不可用，请检查TAVILY_API_KEY配置")
        elif self.backend == "serpapi" and "serpapi" not in self.available_backends:
            logger.warning("SerpApi不可用，请检查SERPAPI_API_KEY配置")
        elif self.backend not in ["tavily", "serpapi", "hybrid"]:
            logger.warning("不支持的搜索后端，将使用hybrid模式")
            self.backend = "hybrid"

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行搜索

        Args:
            parameters: 包含input参数的字典

        Returns:
            搜索结果
        """
        query = parameters.get("input", "").strip()
        if not query:
            return "错误：搜索查询不能为空"

        logger.info("正在执行搜索: %s", query)

        try:
            if self.backend == "hybrid":
                return self._search_hybrid(query)
            elif self.backend == "tavily":
                if "tavily" not in self.available_backends:
                    return self._get_api_config_message()
                return self._search_tavily(query)
            elif self.backend == "serpapi":
                if "serpapi" not in self.available_backends:
                    return self._get_api_config_message()
                return self._search_serpapi(query)
            else:
                return self._get_api_config_message()
        except Exception as e:
            return f"搜索时发生错误: {str(e)}"

    def _search_hybrid(self, query: str) -> str:
        """混合搜索 - 智能选择最佳搜索源"""
        # 检查是否有可用的搜索源
        if not self.available_backends:
            return self._get_api_config_message()

        # 优先使用Tavily（AI优化的搜索）
        if "tavily" in self.available_backends:
            try:
                logger.info("使用Tavily进行AI优化搜索")
                return self._search_tavily(query)
            except Exception as e:
                logger.warning("Tavily搜索失败: %s", e)
                # 如果Tavily失败，尝试SerpApi
                if "serpapi" in self.available_backends:
                    logger.info("切换到SerpApi搜索")
                    return self._search_serpapi(query)

        # 如果Tavily不可用，使用SerpApi
        elif "serpapi" in self.available_backends:
            try:
                logger.info("使用SerpApi进行Google搜索")
                return self._search_serpapi(query)
            except Exception as e:
                logger.warning("SerpApi搜索失败: %s", e)

        # 如果都失败了，返回API配置提示
        return "所有搜索源都失败了，请检查网络连接和API密钥配置"

    def _search_tavily(self, query: str) -> str:
        """使用Tavily搜索"""
        response = self.tavily_client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=3
        )

        result = f"Tavily AI搜索结果：{response.get('answer', '未找到直接答案')}\n\n"

        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:200]}...\n"
            result += f"    来源: {item.get('url', '')}\n\n"

        return result

    def _search_serpapi(self, query: str) -> str:
        """使用SerpApi搜索"""
        try:
            from serpapi import SerpApiClient
        except ImportError:
            return "错误：SerpApi未安装，请运行 pip install serpapi"

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        result_text = "SerpApi Google搜索结果：\n\n"

        # 智能解析：优先寻找最直接的答案
        if "answer_box" in results and "answer" in results["answer_box"]:
            result_text += f"直接答案：{results['answer_box']['answer']}\n\n"

        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            result_text += f"知识图谱：{results['knowledge_graph']['description']}\n\n"

        if "organic_results" in results and results["organic_results"]:
            result_text += "相关结果：\n"
            for i, res in enumerate(results["organic_results"][:3], 1):
                result_text += f"[{i}] {res.get('title', '')}\n"
                result_text += f"    {res.get('snippet', '')}\n"
                result_text += f"    来源: {res.get('link', '')}\n\n"
            return result_text

        return f"对不起，没有找到关于 '{query}' 的信息。"

    def _get_api_config_message(self) -> str:
        """获取API配置提示信息"""
        tavily_key = os.getenv("TAVILY_API_KEY")
        serpapi_key = os.getenv("SERPAPI_API_KEY")

        message = "没有可用的搜索源，请检查以下配置：\n\n"

        # 检查Tavily
        message += "1. Tavily API:\n"
        if not tavily_key:
            message += "   环境变量 TAVILY_API_KEY 未设置\n"
            message += "   获取地址: https://tavily.com/\n"
        else:
            try:
                import tavily
                message += "   API密钥已配置，包已安装\n"
            except ImportError:
                message += "   API密钥已配置，但需要安装包: pip install tavily-python\n"

        message += "\n"

        # 检查SerpAPI
        message += "2. SerpAPI:\n"
        if not serpapi_key:
            message += "   环境变量 SERPAPI_API_KEY 未设置\n"
            message += "   获取地址: https://serpapi.com/\n"
        else:
            try:
                import serpapi
                message += "   API密钥已配置，包已安装\n"
            except ImportError:
                message += "   API密钥已配置，但需要安装包: pip install google-search-results\n"

        message += "\n配置方法：\n"
        message += "- 在.env文件中添加: TAVILY_API_KEY=your_key_here\n"
        message += "- 或在环境变量中设置: export TAVILY_API_KEY=your_key_here\n"
        message += "\n配置后重新运行程序。"

        return message

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="input",
                type="string",
                description="搜索查询关键词",
                required=True
            )
        ]

# 便捷函数
def search(query: str, backend: str = "hybrid") -> str:
    """
    便捷的搜索函数

    Args:
        query: 搜索查询关键词
        backend: 搜索后端 ("hybrid", "tavily", "serpapi")

    Returns:
        搜索结果
    """
    tool = SearchTool(backend=backend)
    return tool.truncate_output(tool.run({"input": query}))

# 专用搜索函数
def search_tavily(query: str) -> str:
    """使用Tavily进行AI优化搜索"""
    tool = SearchTool(backend="tavily")
    return tool.truncate_output(tool.run({"input": query}))

def search_serpapi(query: str) -> str:
    """使用SerpApi进行Google搜索"""
    tool = SearchTool(backend="serpapi")
    return tool.truncate_output(tool.run({"input": query}))

def search_hybrid(query: str) -> str:
    """智能混合搜索，自动选择最佳搜索源"""
    tool = SearchTool(backend="hybrid")
    return tool.truncate_output(tool.run({"input": query}))

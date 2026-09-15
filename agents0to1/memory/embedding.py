import os
from typing import List, Optional
from openai import OpenAI
from ..utils.logging import get_logger
from ..core.exceptions import EmbeddingException

"""为embedding提供open ai"""

logger = get_logger(__name__)


_EMBEDDING_PROVIDERS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "openai": (
        "text-embedding-3-small",
        "https://api.openai.com/v1",
        ("OPENAI_API_KEY",),
    ),
    "dashscope": (
        "text-embedding-v3",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ("DASHSCOPE_API_KEY",),
    ),
    "zhipu": (
        "embedding-3",
        "https://open.bigmodel.cn/api/paas/v4",
        ("ZHIPU_API_KEY", "GLM_API_KEY"),
    ),
    "modelscope": (
        "Qwen/Qwen3-Embedding-0.6B",
        "https://api-inference.modelscope.cn/v1/",
        ("MODELSCOPE_API_KEY",),
    ),
    "ollama": (
        "nomic-embed-text",
        "http://localhost:11434/v1",
        ("OLLAMA_EMBEDDING_HOST", "OLLAMA_HOST"),
    ),
}


# 检测顺序
_EMBEDDING_PROVIDER_ORDER = ("openai", "dashscope", "zhipu", "modelscope", "ollama")

class EmbeddingClient:
    """Embedding 客户端 —— 批量把文本转成向量。

    配置优先级(每一项都一样): 显式传参 > EMBEDDING_* 环境变量 > provider 默认值

        EMBEDDING_PROVIDER    openai / dashscope / zhipu / modelscope / ollama
        EMBEDDING_MODEL       text-embedding-3-small / text-embedding-v3 / ...
        EMBEDDING_BASE_URL    可选
        EMBEDDING_API_KEY     可选

    用法:
        c = EmbeddingClient()
        vs = c.embed(["你好", "世界"])     # 一次请求发两条, 不是发两次
        print(c.dim)                       # 768 / 1024 / 1536 ...

    【和 chat 客户端最大的调用习惯差异】必须攒批。
    一次请求发 32 条 vs 发 32 次请求, 差几十倍 —— 不是省几毫秒的问题。
    """
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 32,
        timeout: int = 30,
    ):
        """
        Args:
            model:      模型名, 默认按 provider 取
            api_key:    API 密钥, 默认按 provider 取环境变量
            base_url:   服务地址, 默认按 provider 取
            provider:   openai / dashscope / zhipu / modelscope / ollama,
                        不传就按环境变量自动检测(见 _detect_provider)
            dimensions: 目标维度。只有 OpenAI 的 text-embedding-3-* 这类支持
                        截断的模型才认这个参数, 不支持的服务传了会报 400 ——
                        所以**只在显式传了的时候**才发出去。
            batch_size: 每次 HTTP 请求塞几条。太大可能撞服务端的单请求上限,
                        太小则浪费往返。
            timeout:    单次请求超时(秒)
        """
        if batch_size <= 0:
            raise EmbeddingException(f"batch_size 必须为正整数, 收到 {batch_size}。")

        self.batch_size = batch_size
        self.timeout = timeout
        self.dimensions = dimensions

        # 1. provider: 显式传 > 环境变量 EMBEDDING_PROVIDER > 自动检测
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER") or self._detect_provider()

        # 2. 拿到这家 provider 的默认值和"该去哪找 key"
        default_model, default_url, key_names = _EMBEDDING_PROVIDERS.get(
            self.provider, (None, None, ())
        )
        if default_model is None:
            known = " / ".join(_EMBEDDING_PROVIDER_ORDER)
            raise EmbeddingException(
                f"不认识的 embedding provider: '{self.provider}'。支持的有: {known}。"
            )

        # 3. 逐项解析
        self.model = model or os.getenv("EMBEDDING_MODEL") or default_model
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or self._first_env(key_names)
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or default_url

        if not self.api_key:
            raise EmbeddingException(
                f"provider '{self.provider}' 没有找到 API 密钥。"
                f"请设置 {' / '.join(key_names)} 中任意一个, 或直接设 EMBEDDING_API_KEY。"
            )

        # 4. 维度缓存: 第一次调用后才知道, 拿到了就记住
        self._dim: Optional[int] = None

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.info(
            "Embedding 客户端就绪: provider=%s model=%s base_url=%s",
            self.provider, self.model, self.base_url,
        )

    # ==================== provider 检测 ====================

    @staticmethod
    def _first_env(names: tuple) -> Optional[str]:
        """按顺序找第一个有值的环境变量"""
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return None

    @classmethod
    def _detect_from_base_url(cls, base_url: str) -> Optional[str]:
        """按 base_url 认 provider。认不出来返回 None。"""
        url = (base_url or "").lower()
        if not url:
            return None
        if "api.openai.com" in url:
            return "openai"
        if "dashscope.aliyuncs.com" in url:
            return "dashscope"
        if "bigmodel.cn" in url:
            return "zhipu"
        if "modelscope" in url:
            return "modelscope"
        if ":11434" in url or "ollama" in url:
            return "ollama"
        return None

    @classmethod
    def _detect_provider(cls) -> str:
        """
        自动检测 embedding provider。
        """
        matched = [
            name
            for name in _EMBEDDING_PROVIDER_ORDER
            if cls._first_env(_EMBEDDING_PROVIDERS[name][2])
        ]
        if not matched:
            raise EmbeddingException(
                "没能自动检测出 embedding provider。请显式设置 EMBEDDING_PROVIDER, "
                "或设置对应 provider 的密钥环境变量。\n"
                " 能自动检测的: " + " / ".join(_EMBEDDING_PROVIDER_ORDER) + "\n"
            )

        if len(matched) == 1:
            return matched[0]

        # 命中多个时用 base_url 打破平局 —— 和 chat 那边的处理保持一致:
        # "环境里恰好有这么个 key" 远不如 base_url 能表达意图。
        hinted = cls._detect_from_base_url(os.getenv("EMBEDDING_BASE_URL", ""))
        winner = hinted if hinted in matched else matched[0]
        logger.warning(
            "检测到多个 embedding provider 的密钥: %s。已选择 '%s'%s。",
            "/".join(matched), winner,
            "(按 EMBEDDING_BASE_URL 判断)" if winner == hinted else "(按检测顺序取第一个)",
        )
        return winner


    # ==================== 核心接口 ====================

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入。内部按 batch_size 切片, 一个批次发一次 HTTP。
        Args:
            texts: 待嵌入的文本列表
        Returns:
            与 texts 顺序严格一一对应的向量列表。
        Raises:
            EmbeddingException: 任何失败都抛。本层**不做** fail-open ——
                需要 fail-open 的是记忆层和 Agent 的注入钩子。
        """
        if not texts:
            return []

        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        vectors: List[List[float]] = []
        for index, batch in enumerate(batches, 1):
            vectors.extend(self._embed_batch(batch, index, len(batches)))

        if len(vectors) != len(texts):
            raise EmbeddingException(
                f"返回条数不对: 发了 {len(texts)} 条, 回来 {len(vectors)} 条。"
            )
        return vectors

    def _embed_batch(self, batch: List[str], index: int, total: int) -> List[List[float]]:
        """发一批。返回条数不对时抛异常, 不猜。"""
        params = {"model": self.model, "input": batch}
        # dimensions 只在显式传了的时候才发: 不支持截断的服务看到这个字段会报 400
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions

        try:
            response = self._client.embeddings.create(**params)
        except Exception as e:
            logger.exception("Embedding 请求失败 (provider=%s model=%s)", self.provider, self.model)
            raise EmbeddingException(
                f"Embedding 调用失败: {e}\n"
                f"  provider={self.provider} model={self.model} base_url={self.base_url}"
            ) from e

        # 按 index 排序, 不假设服务端一定按顺序返回
        ordered = sorted(response.data, key=lambda d: d.index)
        vectors = [list(item.embedding) for item in ordered]

        if len(vectors) != len(batch):
            raise EmbeddingException(
                f"第 {index}/{total} 批返回条数不对: 发了 {len(batch)} 条, "
                f"回来 {len(vectors)} 条。"
            )

        self._remember_dim(vectors[0])
        return vectors

    def _remember_dim(self, vector: List[float]) -> None:
        """
        记下维度, 并校验之后每次都对得上。
        同一库里混着两种长度的向量, 余弦相似度算出来是纯噪声, 而且不会报错
        它只会给你错误的结果。所以在最靠近源头的地方就把它拦下来。
        """
        size = len(vector)
        if self._dim is None:
            self._dim = size
            logger.info("Embedding 维度已确定: %d (model=%s)", size, self.model)
            return

        if size != self._dim:
            raise EmbeddingException(
                f"向量维度变了: 之前是 {self._dim}, 这次是 {size}。"
            )

    def embed_one(self, text: str) -> List[float]:
        """单条。内部就是 embed([text])[0]。"""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        """
        向量维度。第一次调用后才知道, 拿到就缓存住。

        还没有向量时抛异常而不是返回 None: 维度是要拿去做校验和建库的,
        一个 None 会一路传到很远的地方才炸。早早报错更好定位。
        """
        if self._dim is None:
            raise EmbeddingException(
                "还不知道向量维度 —— 至少调用一次 embed() 之后才可用。"
            )
        return self._dim

    # ==================== 杂项 ====================

    @property
    def model_id(self) -> str:
        """
        写进向量库的"模型指纹"。

        【为什么带上 provider】不同 provider 的 text-embedding-v3 也不是同一批
        权重, 向量不可比。只记模型名会在换 provider 时漏判。
        """
        return f"{self.provider}/{self.model}"

    def __str__(self) -> str:
        dim = self._dim if self._dim is not None else "?"
        return f"EmbeddingClient(provider={self.provider}, model={self.model}, dim={dim})"

    def __repr__(self) -> str:
        return self.__str__()


# ==================== 验证(指南第二步) ====================
# 需要 key, 所以不放在离线测试里。手动跑:
#
#     python -c "from agents0to1.memory.embedding import EmbeddingClient; \
#                c = EmbeddingClient(); vs = c.embed(['你好', '世界']); \
#                print(len(vs), len(vs[0]), c.dim)"
#
# 期望: 2 <dim> <dim>  —— 第三条读的是缓存, 没有再发请求。
#
# 本地没配任何云端 key 时, 用 ollama 免费跑:
#     ollama pull nomic-embed-text
#     set EMBEDDING_PROVIDER=ollama   (PowerShell: $env:EMBEDDING_PROVIDER="ollama")

"""
embedder.py — Embedding 模型统一封装
════════════════════════════════════════
bge-m3 (is_hybrid=True)  : 稠密 + 稀疏混合检索，需要 FlagEmbedding
其他模型 (is_hybrid=False): 仅稠密检索，使用 SentenceTransformer

外部接口：
  model.encode(text_or_list)            → 稠密向量（单串→1D, 列表→2D）
  model.encode_hybrid(list_of_str)      → (dense_matrix, sparse_list)
  model.is_hybrid                       → bool
  sparse_dot(q_sparse, d_sparse)        → float  混合得分的稀疏部分
"""
import numpy as np
from config import get_logger

logger = get_logger("Embedder")


def sparse_dot(q: dict, d: dict) -> float:
    """稀疏向量内积（token_id 字符串 → 权重）"""
    if not q or not d:
        return 0.0
    return sum(float(q.get(k, 0.0)) * float(v) for k, v in d.items())


class EmbeddingModel:
    """
    统一封装，屏蔽两种后端差异。
    classifier / memory_store / reply_bot 所有地方只接触这个类。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._backend   = None
        self.is_hybrid  = False
        self._load(model_name)

    # ══════════════════════════════════════════════
    #  初始化
    # ══════════════════════════════════════════════
    def _load(self, model_name: str):
        if "bge-m3" in model_name.lower():
            try:
                from FlagEmbedding import BGEM3FlagModel
                logger.info(f"加载 BGE-M3 混合检索模型: {model_name}")
                self._backend  = BGEM3FlagModel(model_name, use_fp16=True)
                self.is_hybrid = True
                logger.info("✅ BGE-M3 就绪（稠密 + 稀疏混合检索）")
                return
            except ImportError:
                logger.warning("⚠️ FlagEmbedding 未安装，回退到 SentenceTransformer（仅稠密向量）")
                logger.warning("   安装命令: pip install FlagEmbedding")

        from sentence_transformers import SentenceTransformer
        logger.info(f"加载 SentenceTransformer 模型: {model_name}")
        self._backend  = SentenceTransformer(model_name)
        self.is_hybrid = False

    # ══════════════════════════════════════════════
    #  稠密编码（对外主接口，向后兼容所有旧调用）
    # ══════════════════════════════════════════════
    def encode(self, sentences, batch_size: int = 64,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = True) -> np.ndarray:
        """
        单字符串 → 1D float32 array (dim,)
        列表     → 2D float32 array (N, dim)
        """
        single = isinstance(sentences, str)
        texts  = [sentences] if single else list(sentences)

        if self.is_hybrid:
            out  = self._backend.encode(
                texts, batch_size=batch_size,
                return_dense=True, return_sparse=False, return_colbert_vecs=False,
            )
            vecs = np.array(out["dense_vecs"], dtype=np.float32)
        else:
            vecs = np.array(
                self._backend.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    normalize_embeddings=False,
                ),
                dtype=np.float32,
            )

        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs  = vecs / np.where(norms == 0, 1.0, norms)

        return vecs[0] if single else vecs

    # ══════════════════════════════════════════════
    #  混合编码（稠密 + 稀疏）
    # ══════════════════════════════════════════════
    def encode_hybrid(self, texts: list[str],
                      batch_size: int = None) -> tuple[np.ndarray, list[dict]]:
        """
        返回 (dense_matrix, sparse_list)
          dense_matrix : (N, dim) float32，已 L2 归一化
          sparse_list  : list of {token_id_str: weight_float}
                         非 M3 时返回 [{} * N]，退化为纯稠密检索
        """
        if not texts:
            dim = 1024 if self.is_hybrid else self.dim()
            return np.zeros((0, dim), dtype=np.float32), []

        if batch_size is None:
            batch_size = 12 if self.is_hybrid else 64

        if self.is_hybrid:
            out    = self._backend.encode(
                texts, batch_size=batch_size,
                return_dense=True, return_sparse=True, return_colbert_vecs=False,
            )
            dense  = np.array(out["dense_vecs"], dtype=np.float32)
            norms  = np.linalg.norm(dense, axis=1, keepdims=True)
            dense  = dense / np.where(norms == 0, 1.0, norms)
            sparse = [{str(k): float(v) for k, v in w.items()}
                      for w in out["lexical_weights"]]
        else:
            dense  = self.encode(texts, batch_size=batch_size, normalize_embeddings=True)
            if dense.ndim == 1:
                dense = dense[np.newaxis, :]
            sparse = [{} for _ in texts]

        return dense, sparse

    # ══════════════════════════════════════════════
    #  工具
    # ══════════════════════════════════════════════
    def dim(self) -> int:
        if self.is_hybrid:
            return 1024
        return self._backend.get_sentence_embedding_dimension()

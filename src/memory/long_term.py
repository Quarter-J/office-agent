"""
长期记忆模块

基于 SQLite 存储用户事实和交互摘要，
基于 FAISS 向量索引实现语义检索。
"""

import json
from pathlib import Path
from typing import Optional
from src.storage.database import insert, query, upsert
from src.config.settings import settings
from src.utils.logger import logger


class LongTermMemory:
    """用户长期记忆：偏好、行为模式、交互摘要"""

    def __init__(self):
        self._vector_store = None
        self._index_dirty = True
        self._vector_path = Path(settings.memory_vector_store_path)

    # ============================================================
    # 事实管理（SQLite）
    # ============================================================

    def save_fact(self, user_id: str, category: str, key: str,
                  value: str, source_intent: str = "", confidence: float = 0.8):
        """保存或更新一条用户事实"""
        upsert("memory_facts", {
            "user_id": user_id,
            "category": category,
            "fact_key": key,
            "fact_value": value,
            "source_intent": source_intent,
            "confidence": confidence,
        }, ["user_id", "category", "fact_key"])
        self._index_dirty = True
        logger.info(f"[Memory] 保存事实: {user_id}/{category}/{key}={value}")

    def get_facts(self, user_id: str, category: str = None) -> list[dict]:
        """获取用户事实"""
        if category:
            return query("memory_facts",
                         where="user_id = ? AND category = ?",
                         params=(user_id, category),
                         order_by="access_count DESC, confidence DESC")
        return query("memory_facts",
                     where="user_id = ?",
                     params=(user_id,),
                     order_by="access_count DESC, confidence DESC")

    def search_facts(self, user_id: str, query_text: str,
                     top_k: int = 5) -> list[dict]:
        """语义检索用户事实"""
        # 尝试 FAISS 向量检索
        results = self._vector_search(user_id, query_text, top_k)
        if results:
            return results

        # 回退：SQLite LIKE 模糊匹配
        return self._fallback_search(user_id, query_text, top_k)

    def save_interaction_summary(self, user_id: str, summary: str,
                                 intent: str = "",
                                 input_snapshot: dict = None,
                                 output_snapshot: dict = None):
        """保存交互摘要"""
        insert("interaction_summaries", {
            "user_id": user_id,
            "intent": intent,
            "summary": summary,
            "input_snapshot": json.dumps(input_snapshot, ensure_ascii=False) if input_snapshot else None,
            "output_snapshot": json.dumps(output_snapshot, ensure_ascii=False) if output_snapshot else None,
        })
        logger.info(f"[Memory] 保存交互摘要: {user_id}/{intent}")

    # ============================================================
    # FAISS 向量检索
    # ============================================================

    def _get_embeddings(self):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            request_timeout=settings.embedding_timeout_seconds,
            max_retries=0,
        )

    def _build_index(self, user_id: str) -> bool:
        """从 SQLite 构建 FAISS 索引"""
        try:
            from langchain_community.vectorstores import FAISS

            facts = self.get_facts(user_id)
            if not facts:
                return False

            texts = []
            metadatas = []
            for f in facts:
                text = f"{f['category']}: {f['fact_key']} = {f['fact_value']}"
                texts.append(text)
                metadatas.append({
                    "fact_id": f["id"],
                    "category": f["category"],
                    "fact_key": f["fact_key"],
                    "fact_value": f["fact_value"],
                    "confidence": f["confidence"],
                })

            embeddings = self._get_embeddings()
            self._vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

            save_path = self._vector_path / user_id
            save_path.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(save_path))
            self._index_dirty = False
            logger.info(f"[Memory] FAISS 索引已构建: {len(texts)} 条事实")
            return True
        except ImportError:
            logger.warning("[Memory] FAISS 依赖缺失，使用模糊匹配回退")
            return False
        except Exception as e:
            logger.warning(f"[Memory] 构建索引失败: {e}")
            return False

    def _load_index(self, user_id: str) -> bool:
        """加载 FAISS 索引（懒加载 + 按需重建）"""
        try:
            from langchain_community.vectorstores import FAISS

            save_path = self._vector_path / user_id
            if not save_path.exists():
                return self._build_index(user_id)

            if self._vector_store is not None and not self._index_dirty:
                return True

            embeddings = self._get_embeddings()
            self._vector_store = FAISS.load_local(
                str(save_path), embeddings,
                allow_dangerous_deserialization=True,
            )
            self._index_dirty = False
            return True
        except Exception:
            return self._build_index(user_id)

    def _vector_search(self, user_id: str, query_text: str,
                       top_k: int) -> list[dict]:
        """FAISS 向量检索"""
        if not self._load_index(user_id):
            return []
        try:
            docs_with_scores = self._vector_store.similarity_search_with_score(
                query_text, k=top_k
            )
            results = []
            for doc, score in docs_with_scores:
                # 增加访问计数
                fact_id = doc.metadata.get("fact_id")
                if fact_id:
                    self._increment_access(fact_id)
                results.append({
                    "category": doc.metadata.get("category", ""),
                    "fact_key": doc.metadata.get("fact_key", ""),
                    "fact_value": doc.metadata.get("fact_value", ""),
                    "confidence": doc.metadata.get("confidence", 0.8),
                    "score": round(float(score), 4),
                })
            return results
        except Exception as e:
            logger.warning(f"[Memory] 向量检索失败: {e}")
            return []

    def _increment_access(self, fact_id: int):
        """增加事实的访问计数"""
        try:
            from src.storage.database import get_connection
            conn = get_connection()
            try:
                conn.execute(
                    "UPDATE memory_facts SET access_count = access_count + 1 WHERE id = ?",
                    (fact_id,)
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def _fallback_search(self, user_id: str, query_text: str,
                         top_k: int) -> list[dict]:
        """SQLite LIKE 模糊匹配回退"""
        all_facts = self.get_facts(user_id)
        keywords = [w for w in query_text if len(w) > 1]  # 简单分字
        scored = []
        for f in all_facts:
            text = f"{f['fact_key']} {f['fact_value']}".lower()
            hits = sum(1 for kw in keywords if kw in text)
            if hits > 0:
                scored.append((hits, f))
        scored.sort(key=lambda x: -x[0])
        return [item[1] for item in scored[:top_k]]

    def rebuild_index(self, user_id: str):
        """手动重建 FAISS 索引"""
        self._index_dirty = True
        self._vector_store = None
        self._build_index(user_id)

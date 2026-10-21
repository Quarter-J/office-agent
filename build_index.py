"""
构建知识库向量索引

从知识库目录中的文档构建 FAISS 向量索引，并抽取实体写入结构化实体库，
启用 RAG 语义检索与精确查找功能。

用法：
  python build_index.py              # 构建索引
  python build_index.py --rebuild    # 强制重建（清空旧索引与实体表）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag.indexer import build_index
from src.utils.logger import logger


def main():
    rebuild = "--rebuild" in sys.argv

    if rebuild:
        import shutil
        from src.config.settings import settings
        from src.storage.database import get_connection

        vector_path = Path(settings.vector_store_path)
        if vector_path.exists():
            shutil.rmtree(vector_path)
            logger.info("已删除旧索引")

        conn = get_connection()
        try:
            conn.execute("DELETE FROM entities")
            conn.commit()
        finally:
            conn.close()
        logger.info("已清空实体表")

    success = build_index()

    if success:
        print()
        print("=" * 50)
        print("  ✅ 向量索引构建成功！")
        print("  知识库文档已完成解析、分块和向量化")
        print("  现在可以使用 RAG 语义检索和实体精确查找了")
        print("=" * 50)
        print()
        print("  试试在对话中输入：")
        print('    "在知识库中搜索..."')
        print('    "张三的电话是多少"')
        print()
    else:
        print()
        print("=" * 50)
        print("  ⚠️  向量索引构建失败")
        print("  请检查：")
        print("  1. data/knowledge_base 目录是否有文档")
        print("  2. 是否安装了依赖：pip install langchain-community faiss-cpu")
        print("  3. 嵌入端点是否可用（EMBEDDING_BASE_URL）")
        print("=" * 50)
        print()


if __name__ == "__main__":
    main()

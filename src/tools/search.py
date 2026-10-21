"""
信息检索工具集

支持：语义向量检索（FAISS）/ RAG 问答 / 实体精确查找 / 网页搜索 / 添加知识库文档
"""

import json
import re
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from src.config.settings import settings
from src.utils.logger import logger
from src.rag.validation import TopK, validate_top_k
from src.utils.execution import timed_tool, read_request, is_timeout, timeout_result


# ============================================================
# 知识库检索
# ============================================================

@tool
@timed_tool()
def knowledge_base_search(query: str, top_k: TopK = 3) -> str:
    """在本地知识库中检索文档片段（语义向量优先，关键词降级）

    Args:
        query: 检索关键词（自然语言）
        top_k: 返回最相关的文档数量，默认3条

    Returns:
        检索到的知识库文档片段
    """
    result = _retrieve(query, top_k)
    result["results"] = [{**hit, "content": hit["content"][:500]} for hit in result["results"]]
    return json.dumps(result, ensure_ascii=False)


def _retrieve(query: str, top_k: int = 3) -> dict:
    """One retrieval policy for snippets and RAG: vector first, keyword fallback."""
    try:
        validate_top_k(top_k)
        if not query.strip():
            raise ValueError("检索问题不能为空")
    except (ValueError, AttributeError) as exc:
        return {"success": False, "status": "error", "results": [], "error": str(exc)}
    vector_error = None
    vector_timed_out = False
    try:
        from src.rag import vector_store
        hits = vector_store.search(query, top_k=top_k, raise_errors=True)
        if hits:
            results = [
                {
                    "rank": i + 1,
                    "source": vector_store.source_label(h.meta),
                    "score": round(h.score, 4),
                    "content": h.text,
                }
                for i, h in enumerate(hits)
            ]
            logger.info(f"语义检索完成: query={query}, results={len(results)}")
            return {"success": True, "status": "ok", "method": "vector", "results": results}
    except Exception as e:
        vector_error = str(e)
        vector_timed_out = is_timeout(e)
        logger.warning(f"语义检索失败，降级关键词检索: {e}")

    result = json.loads(_keyword_search(query, top_k))
    if vector_error:
        result["fallback_reason"] = vector_error
        if not result.get("results"):
            result.update(success=False, status="error",
                          error="语义检索不可用，关键词检索也未能返回资料：" + vector_error)
            if vector_timed_out:
                result.update(status="timeout", error_type="timeout", retryable=False)
    return result


def _keyword_search(query: str, top_k: int = 3) -> str:
    """降级：jieba 关键词检索"""
    try:
        validate_top_k(top_k)
        import jieba

        kb_dir = Path(settings.knowledge_base_dir)
        if not kb_dir.exists() or not any(kb_dir.iterdir()):
            return json.dumps({
                "success": True,
                "status": "empty",
                "method": "keyword",
                "results": [],
                "message": "知识库为空，请先添加文档",
            }, ensure_ascii=False)

        query_words = {word for word in jieba.cut_for_search(query) if word.strip()}

        documents = []
        for file_path in kb_dir.rglob("*.*"):
            if file_path.suffix in [".txt", ".md", ".py", ".java", ".yaml", ".yml", ".json", ".xml"]:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
                    for p in paragraphs:
                        if len(p) > 20:
                            documents.append({
                                "file": str(file_path.relative_to(kb_dir)),
                                "content": p,
                            })
                except Exception:
                    continue

        scored_docs = []
        for doc in documents:
            doc_words = set(jieba.cut_for_search(doc["content"]))
            intersection = query_words & doc_words
            score = len(intersection) / len(query_words) if query_words else 0
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = [
            {
                "rank": i + 1,
                "source": doc["file"],
                "score": round(score, 4),
                "content": doc["content"],
            }
            for i, (score, doc) in enumerate(scored_docs[:top_k])
        ]
        return json.dumps({"success": True, "status": "ok" if results else "empty",
                           "method": "keyword", "results": results}, ensure_ascii=False)
    except ImportError:
        return json.dumps({"success": False, "status": "error", "results": [], "error": "缺少 jieba 依赖，请安装: pip install jieba"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"关键词检索失败: {e}")
        return json.dumps({"success": False, "status": "error", "results": [], "error": str(e)}, ensure_ascii=False)


@tool
@timed_tool()
def rag_search(query: str, top_k: TopK = 3) -> str:
    """RAG 智能检索：语义检索知识库 + LLM 生成带引用来源的回答

    Args:
        query: 用户问题（自然语言）
        top_k: 检索的文档片段数量

    Returns:
        LLM 生成的回答（含引用来源）
    """
    result = _retrieve(query, top_k)
    fragments = result["results"]
    sources = list(dict.fromkeys(fragment["source"] for fragment in fragments))
    result.update(sources=sources, answer="")
    if result["status"] != "ok":
        if result["status"] == "empty":
            result["answer"] = "知识库中未找到与您问题相关的资料。"
        return json.dumps(result, ensure_ascii=False)
    try:
        from src.tools.prompts import RAG_QA_TEMPLATE
        from src.api.llm import get_llm
        context_text = "\n\n".join(f"【来源：{f['source']}】\n{f['content']}" for f in fragments)

        llm = get_llm()
        prompt = RAG_QA_TEMPLATE.format_messages(context=context_text, query=query)
        from langchain_core.messages import SystemMessage
        prompt[1:2] = [SystemMessage(name="context_reference", content=f"【来源：{fragment['source']}】\n{fragment['content']}")
                       for fragment in fragments]
        response = llm.invoke(prompt)
        answer = response.content.strip()
        if not answer:
            raise ValueError("模型未返回回答")

        logger.info(f"RAG 检索完成: query={query}, sources={len(sources)}")
        result["answer"] = answer
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
        # Keep retrieved evidence if answer generation fails; callers can show it.
        result.update(success=False, status="error", error=str(e))
        if is_timeout(e):
            result.update(status="timeout", error_type="timeout", retryable=False)
    return json.dumps(result, ensure_ascii=False)


@tool
@timed_tool()
def lookup_entity(name: str, entity_type: str = "person", field: str = None) -> str:
    """在知识库结构化实体库中精确查找某人/组织的联系方式（电话、邮箱等）

    适用于"张三的电话/邮箱是多少"这类精确查询。语义检索不擅长这类低语义事实。
    命中多个同名实体会返回候选列表要求用户确认，绝不静默挑选。

    Args:
        name: 姓名或组织名，如 张三
        entity_type: 实体类型，默认 person
        field: 可选，指定要查的属性，如 phone / email

    Returns:
        匹配结果（唯一 / 多候选 / 未找到）
    """
    try:
        from src.rag import entity_store

        matches = entity_store.lookup_entity(entity_type, name, field)
        if matches:
            status = "unique" if len(matches) == 1 else "ambiguous"
            return json.dumps({
                "success": True,
                "status": status,
                "matches": matches,
                "message": _format_matches(matches),
            }, ensure_ascii=False)

        fuzzy = entity_store.search_entities(name, top_k=3)
        return json.dumps({
            "success": True,
            "status": "not_found",
            "matches": [],
            "fuzzy": fuzzy,
            "message": _format_not_found(name, fuzzy),
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"实体查找失败: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def _fmt_attrs(attrs: dict) -> str:
    parts = []
    if attrs.get("phone"):
        parts.append("电话 " + " / ".join(attrs["phone"]))
    if attrs.get("email"):
        parts.append("邮箱 " + " / ".join(attrs["email"]))
    return "，".join(parts) or "（无联系方式）"


def _format_matches(matches: list) -> str:
    if len(matches) == 1:
        m = matches[0]
        return f"「{m['label']}」：{_fmt_attrs(m['attrs'])}（来源：{m['source_file']}）"
    lines = [f"找到 {len(matches)} 个「{matches[0]['name']}」，需用户确认是哪一个："]
    for i, m in enumerate(matches, 1):
        src = f"（来源：{m['source_file']}）" if m.get("source_file") else ""
        lines.append(f"{i}. {m['label']}：{_fmt_attrs(m['attrs'])}{src}")
    return "\n".join(lines)


def _format_not_found(name: str, fuzzy: list) -> str:
    if fuzzy:
        lines = [f"未找到与「{name}」完全匹配的联系人。你可能要找的是："]
        for f in fuzzy:
            lines.append(f"- {f['label']}：{_fmt_attrs(f['attrs'])}")
        return "\n".join(lines)
    return f"未找到「{name}」的联系方式。请确认姓名是否正确，或该信息是否已加入知识库。"


# ============================================================
# 网页搜索
# ============================================================

@tool
@timed_tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet using Baidu search engine.

    Args:
        query: Search keywords
        max_results: Maximum number of results, default 5

    Returns:
        Search results with title, snippet, and URL
    """
    try:
        import requests
        import re
        from bs4 import BeautifulSoup
        from urllib.parse import quote

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        url = f"https://www.baidu.com/s?wd={quote(query)}&rn={max_results}"
        with requests.Session() as session:
            session.trust_env = False
            resp = read_request(session.get, url, headers=headers)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")

        results = []
        for item in soup.select(".result, .c-container"):
            if len(results) >= max_results:
                break

            title_tag = item.select_one("h3 a, .t a")
            title = title_tag.get_text(strip=True) if title_tag else ""

            snippet_tag = item.select_one(".c-abstract, .content-right_8Zs40, .c-span-last")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            link = title_tag.get("href", "") if title_tag else ""

            if title:
                results.append({
                    "index": len(results) + 1,
                    "title": title,
                    "snippet": snippet[:200] if snippet else "",
                    "url": link,
                })

        logger.info(f"Baidu search completed: query={query}, results={len(results)}")
        return json.dumps({"success": True, "results": results}, ensure_ascii=False)

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return json.dumps({
            "success": False,
            "error": f"Missing dependency: {e}. Install: pip install requests beautifulsoup4",
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Baidu search failed: {e}")
        if is_timeout(e):
            return json.dumps(timeout_result("网页搜索超时，有限重试后仍未获得结果。"), ensure_ascii=False)
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# ============================================================
# 添加知识库文档
# ============================================================

@tool
@timed_tool(effect="write")
def add_to_knowledge_base(file_path: str, content: Optional[str] = None) -> str:
    """向知识库添加文档

    Args:
        file_path: 文件路径（相对于知识库目录或绝对路径）
        content: 文件内容（如果为空则读取指定文件）

    Returns:
        添加结果
    """
    try:
        kb_dir = Path(settings.knowledge_base_dir)
        kb_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(file_path)
        if source_path.is_absolute():
            if not source_path.exists():
                return json.dumps({
                    "success": False,
                    "error": f"文件不存在: {file_path}",
                }, ensure_ascii=False)
            dest_path = kb_dir / source_path.name
            dest_path.write_text(
                content or source_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            if content:
                dest_path = kb_dir / file_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_text(content, encoding="utf-8")
            else:
                return json.dumps({
                    "success": False,
                    "error": "请提供文件内容",
                }, ensure_ascii=False)

        logger.info(f"知识库添加成功: {dest_path}")

        # 增量索引（失败不影响"已添加"的结果）
        try:
            from src.rag.indexer import incremental_index
            incremental_index([str(dest_path)])
        except Exception as e:
            logger.warning(f"增量索引失败（可稍后运行 python build_index.py）: {e}")

        return json.dumps({
            "success": True,
            "message": f"已添加到知识库: {dest_path.relative_to(kb_dir)}",
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"知识库添加失败: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

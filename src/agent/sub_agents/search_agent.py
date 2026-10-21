"""
Search Agent - 搜索专家

独立负责：
  1. RAG 知识库检索（FAISS 语义检索 → 降级 jieba）
  2. 添加知识库文档
"""

import json
from src.agent.state import AgentState
from src.agent.context import latest_user_input
from src.utils.logger import logger


def search_agent(state: AgentState) -> dict:
    """搜索子 Agent：处理知识库检索任务"""
    logger.info("[SearchAgent] 开始处理知识库检索任务")

    user_input = latest_user_input(state)

    if not user_input:
        return {"final_output": "未找到用户输入", "task_status": "failed"}

    intent = state.get("intent", "")

    if intent == "web_search":
        return _handle_web_search(state, user_input)
    elif intent == "add_knowledge":
        return _handle_add_knowledge(state, user_input)
    else:
        return _handle_knowledge_search(state, user_input)


def _handle_knowledge_search(state: AgentState, user_input: str) -> dict:
    query = state.get("search_query", "")

    if not query:
        query = user_input

    if not query:
        return {
            "final_output": "⚠️ 请提供要检索的关键词",
            "task_status": "pending",
            "needs_clarification": True,
            "clarification_question": "请提供要检索的关键词",
        }

    logger.info(f"[SearchAgent] 知识库检索: {query}")

    return _try_rag_search(query)


def _try_rag_search(query: str):
    """尝试 RAG 语义检索（FAISS + LLM）"""
    try:
        from src.tools.search import rag_search
        raw = rag_search.invoke({"query": query})
        result = json.loads(raw)

        if result.get("status") == "empty":
            return {"final_output": result["answer"], "task_status": "success", "knowledge_results": []}
        if result.get("status") == "ok" and result.get("answer"):
            answer = result["answer"]
            sources = result.get("sources", [])
            if sources:
                answer += f"\n\n📚 **来源**: {', '.join(sources)}"
            return {"final_output": answer, "task_status": "success", "knowledge_results": result["results"]}
        if result.get("results"):
            return {"final_output": "回答生成失败，已找到以下资料：\n\n" + _format_hits(result["results"]),
                    "knowledge_results": result["results"], "task_status": "timeout" if result.get("status") == "timeout" else "failed",
                    "error": result.get("error")}
        return {"final_output": f"❌ 检索失败: {result.get('error', '未知错误')}",
                "task_status": "timeout" if result.get("status") == "timeout" else "failed"}
    except Exception as e:
        logger.warning(f"[SearchAgent] RAG 检索失败: {e}")

        return {"final_output": f"❌ 检索失败: {e}", "task_status": "failed"}


def _format_hits(results: list) -> str:
    return "\n\n".join(f"[{r['rank']}] 📁 **{r['source']}** (分数: {r['score']:.3f})\n"
                       f"{r['content'][:500]}" for r in results)


def _try_keyword_search(query: str):
    """降级：传统 jieba 关键词检索"""
    try:
        from src.tools.search import knowledge_base_search
        raw = knowledge_base_search.invoke({"query": query})
        result = json.loads(raw)

        if not result.get("success"):
            return {"final_output": f"❌ 检索失败: {result.get('error', '未知错误')}", "task_status": "failed"}

        results = result.get("results", [])
        message = result.get("message", "")
        if message:
            return {"final_output": f"📚 {message}", "task_status": "success"}
        if not results:
            return None

        return {
            "final_output": f"📚 检索「{query}」结果:\n\n" + _format_hits(results),
            "knowledge_results": results,
            "task_status": "success",
        }
    except Exception as e:
        logger.warning(f"[SearchAgent] 关键词检索失败: {e}")

    return None


def _handle_web_search(state: AgentState, user_input: str) -> dict:
    """网页搜索：调用 DuckDuckGo 搜索并用 LLM 总结结果"""
    query = state.get("search_query", "")
    if not query:
        query = user_input

    if not query:
        return {
            "final_output": "\u26a0\ufe0f 请提供搜索关键词",
            "task_status": "pending",
            "needs_clarification": True,
            "clarification_question": "请提供搜索关键词",
        }

    logger.info(f"[SearchAgent] 网页搜索: {query}")

    try:
        from src.tools.search import web_search
        raw = web_search.invoke({"query": query, "max_results": 5})
        result = json.loads(raw)

        if not result.get("success"):
            return {
                "final_output": f"\u274c 网页搜索失败: {result.get('error', '未知错误')}",
                "task_status": "timeout" if result.get("status") == "timeout" else "failed",
            }

        results = result.get("results", [])
        if not results:
            return {
                "final_output": f"\U0001f50d 未找到与「{query}」相关的网页结果",
                "task_status": "success",
            }

        # Try LLM summary
        try:
            from src.api.llm import get_llm
            from src.tools.prompts import WEB_SEARCH_SUMMARY_TEMPLATE
            llm = get_llm()

            search_text = ""
            for r in results:
                search_text += f"[{r['index']}] {r['title']}\n   {r['snippet']}\n   链接: {r['url']}\n\n"

            prompt = WEB_SEARCH_SUMMARY_TEMPLATE.format_messages(
                search_results=search_text,
                query=query,
            )
            response = llm.invoke(prompt)
            answer = response.content

            sources = [r["url"] for r in results[:3]]
            source_text = "\n".join([f"- {url}" for url in sources])
            final = f"{answer}\n\n---\n\U0001f517 **来源**:\n{source_text}"
            return {"final_output": final, "task_status": "success"}

        except Exception as e:
            logger.warning(f"[SearchAgent] LLM summary failed, returning raw: {e}")
            formatted = []
            for r in results:
                formatted.append(f"[{r['index']}] **{r['title']}**\n   {r['snippet']}\n   \U0001f517 {r['url']}")
            return {
                "final_output": f"\U0001f50d 搜索「{query}」结果:\n\n" + "\n\n".join(formatted),
                "task_status": "success",
            }

    except Exception as e:
        logger.error(f"[SearchAgent] 网页搜索异常: {e}")
        return {"final_output": f"\u274c 网页搜索异常: {e}", "task_status": "failed"}


def _handle_add_knowledge(state: AgentState, user_input: str) -> dict:
    file_name = state.get("knowledge_file_name", "")
    content = state.get("knowledge_file_content", "")

    if not file_name or not content:
        try:
            from src.api.llm import get_llm
            from src.tools.prompts import ADD_KNOWLEDGE_EXTRACT_TEMPLATE
            llm = get_llm()
            prompt = ADD_KNOWLEDGE_EXTRACT_TEMPLATE.format_messages(user_input=user_input)
            response = llm.invoke(prompt)
            info = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            file_name = file_name or info.get("file_name", "untitled.md")
            content = content or info.get("content", "")
        except:
            pass

    if not content:
        return {
            "final_output": "⚠️ 信息不完整，请提供要添加的文档内容",
            "task_status": "pending",
            "needs_clarification": True,
            "clarification_question": "请提供要添加到知识库的文档内容",
        }

    try:
        from src.tools.search import add_to_knowledge_base
        raw = add_to_knowledge_base.invoke({"file_path": file_name or "untitled.md", "content": content})
        result = json.loads(raw)
        if result.get("status") in ("conflict", "unknown", "timeout"):
            return {"final_output": result["error"], "error": result["error"], "task_status": result["status"]}
        if result.get("success"):
            return {"final_output": f"📝 {result['message']}", "task_status": "success"}
        else:
            return {"final_output": f"❌ 添加失败: {result.get('error', '未知错误')}", "task_status": "failed"}
    except Exception as e:
        logger.error(f"[SearchAgent] 异常: {e}")
        return {"final_output": f"❌ 添加异常: {e}", "task_status": "failed"}

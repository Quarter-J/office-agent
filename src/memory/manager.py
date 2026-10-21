"""
记忆管理器

统一门面，供 Graph 节点调用。
管理短期记忆（Redis）和长期记忆（SQLite + FAISS）的读写。
"""

import json
import threading
from dataclasses import dataclass, field
from src.config.settings import settings
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.prompts import FACT_EXTRACTION_TEMPLATE, SUMMARY_COMPRESSION_TEMPLATE
from src.utils.logger import logger
from src.api.context import summary_chunks


@dataclass
class MemoryContext:
    """加载后的记忆上下文"""
    recent_turns: list = field(default_factory=list)
    conversation_summary: str = ""
    relevant_facts: list = field(default_factory=list)
    user_preferences: list = field(default_factory=list)
    formatted_context: str = ""


class MemoryManager:
    """双层记忆系统统一管理器"""

    def __init__(self):
        self.short_term = ShortTermMemory(
            redis_url=settings.redis_url,
            ttl=settings.memory_short_term_ttl,
            max_turns=settings.memory_short_term_max_turns,
        )
        self.long_term = LongTermMemory()
        self._enabled = settings.memory_enabled
        logger.info("[Memory] MemoryManager 初始化完成")

    def load_context(self, user_id: str, session_id: str,
                     current_input: str) -> MemoryContext:
        """图入口：加载记忆上下文"""
        ctx = MemoryContext()
        if not self._enabled:
            return ctx

        try:
            # 1. 短期记忆：近期对话 + 摘要
            ctx.recent_turns = self.short_term.get_conversation(session_id)
            ctx.conversation_summary = self.short_term.get_summary(session_id)

            # 2. 长期记忆：语义匹配的事实
            ctx.relevant_facts = self.long_term.search_facts(
                user_id, current_input, top_k=settings.memory_long_term_top_k
            )

            # 3. 长期记忆：用户偏好
            ctx.user_preferences = self.long_term.get_facts(
                user_id, category="preference"
            )

            # 4. 格式化
            ctx.formatted_context = self._format_context(ctx, current_input)
            logger.info(f"[Memory] 上下文已加载: {len(ctx.recent_turns)} 轮对话, "
                        f"{len(ctx.relevant_facts)} 条相关事实, "
                        f"{len(ctx.user_preferences)} 条偏好")
        except Exception as e:
            logger.warning(f"[Memory] 加载上下文失败: {e}")

        return ctx

    def save_interaction(self, user_id: str, session_id: str,
                         user_input: str, final_output: str,
                         intent: str = ""):
        """图出口：保存交互到记忆"""
        if not self._enabled:
            return

        try:
            # 1. 短期记忆：保存本轮对话
            self.short_term.add_turn(session_id, "user", user_input)
            self.short_term.add_turn(session_id, "assistant", final_output)

            # 2. 摘要压缩
            turn_count = self.short_term.get_turn_count(session_id)
            if turn_count >= settings.memory_summary_trigger_turns:
                self._compress_conversation(session_id)

            # 3. 长期记忆：保存交互摘要
            self.long_term.save_interaction_summary(
                user_id, summary=f"用户: {user_input[:100]} -> 助手: {final_output[:100]}",
                intent=intent,
                input_snapshot={"input": user_input[:200]},
                output_snapshot={"output": final_output[:200]},
            )

            # 4. 异步提取用户事实（不阻塞响应）
            threading.Thread(
                target=self._extract_and_save_facts,
                args=(user_id, user_input, final_output, intent),
                daemon=True,
            ).start()

        except Exception as e:
            logger.warning(f"[Memory] 保存交互失败: {e}")

    def _compress_conversation(self, session_id: str):
        """用 LLM 压缩旧轮次为摘要"""
        try:
            from src.api.llm import get_llm
            from langchain_core.messages import HumanMessage

            turns = self.short_term.get_conversation(session_id)
            if len(turns) < settings.memory_summary_trigger_turns:
                return

            # Preserve whole recent user/assistant turns. Counts in the legacy
            # configuration are message counts, not pairs.
            keep = min(settings.context_recent_turns * 2, settings.memory_summary_trigger_turns - 2)
            count = len(turns) - keep
            while count > 0 and turns[count].get("role") != "user":
                count -= 1
            if count <= 0:
                return
            old_turns = turns[:count]
            conversation_text = "\n".join(
                f"{t['role']}: {t['content']}" for t in old_turns
            )

            llm = get_llm()
            old_summary = self.short_term.get_summary(session_id)
            new_summary = old_summary[-settings.memory_summary_max_chars:]
            # Reserve space for the previous summary, instructions and JSON
            # escaping. All chunks must succeed before removing any history.
            input_budget = settings.llm_context_window_tokens - settings.llm_output_tokens - settings.llm_context_safety_tokens
            chunk_budget = input_budget // 2 - settings.memory_summary_max_chars * 4 - 1024
            for chunk in summary_chunks(conversation_text, min(8000, chunk_budget)):
                prompt = SUMMARY_COMPRESSION_TEMPLATE.format_messages(
                    conversation=chunk, previous_summary=new_summary,
                    max_chars=settings.memory_summary_max_chars)
                response = llm.invoke(prompt)
                if not isinstance(response.content, str) or not response.content.strip():
                    raise ValueError("摘要模型未返回有效文本")
                new_summary = response.content.strip()[:settings.memory_summary_max_chars]
            if self.short_term.commit_summary(session_id, turns, old_summary, new_summary, count):
                logger.info(f"[Memory] 已合并摘要并移除 {count} 条旧消息")
            else:
                logger.info("[Memory] 历史在摘要期间发生变化，本次不提交摘要")
        except Exception as e:
            logger.warning(f"[Memory] 压缩对话失败: {e}")

    def _extract_and_save_facts(self, user_id: str, user_input: str,
                                final_output: str, intent: str):
        """异步：从交互中提取用户事实并保存"""
        try:
            from src.api.llm import get_llm
            from langchain_core.messages import HumanMessage

            llm = get_llm()
            prompt = FACT_EXTRACTION_TEMPLATE.format(
                intent=intent,
                user_input=user_input[:500],
                assistant_output=final_output[:500],
            )
            response = llm.invoke([HumanMessage(content=prompt)])

            facts = self._parse_facts(response.content)
            threshold = settings.memory_fact_confidence_threshold

            for fact in facts:
                if fact.get("confidence", 0) >= threshold:
                    self.long_term.save_fact(
                        user_id=user_id,
                        category=fact["category"],
                        key=fact["key"],
                        value=fact["value"],
                        source_intent=intent,
                        confidence=fact["confidence"],
                    )

            if facts:
                logger.info(f"[Memory] 提取到 {len(facts)} 条事实")
        except Exception as e:
            logger.warning(f"[Memory] 事实提取失败: {e}")

    def _parse_facts(self, text: str) -> list[dict]:
        """解析 LLM 返回的事实 JSON"""
        try:
            # 尝试直接解析
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            # 尝试提取 JSON 数组部分
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return []

    def _format_context(self, ctx: MemoryContext, current_input: str) -> str:
        """将记忆格式化为可注入 prompt 的字符串"""
        parts = []

        # 摘要
        if ctx.conversation_summary:
            summary = ctx.conversation_summary[:settings.memory_summary_max_chars]
            parts.append(f"【近期对话摘要】{summary}")

        # 相关事实
        if ctx.relevant_facts:
            facts_lines = []
            for f in ctx.relevant_facts[:5]:
                facts_lines.append(
                    f"- {f['fact_key'][:100]}: {f['fact_value'][:300]}（置信度: {f.get('confidence', 0.8):.1f}）"
                )
            parts.append("【相关记忆】\n" + "\n".join(facts_lines))

        # 用户偏好（去重：排除已在 relevant_facts 中出现的）
        relevant_keys = {f.get("fact_key") for f in ctx.relevant_facts}
        prefs = [f for f in ctx.user_preferences if f["fact_key"] not in relevant_keys]
        if prefs:
            pref_lines = []
            for f in prefs[:5]:
                pref_lines.append(f"- {f['fact_key'][:100]}: {f['fact_value'][:300]}")
            parts.append("【用户偏好】\n" + "\n".join(pref_lines))

        # 近期对话（最近 3 轮）
        if ctx.recent_turns:
            recent = ctx.recent_turns[-6:]  # 最多 3 轮 = 6 条消息
            turn_lines = [f"{t['role']}: {t['content'][:100]}" for t in recent]
            parts.append("【近期对话】\n" + "\n".join(turn_lines))

        if not parts:
            return ""

        return "\n\n=== 记忆上下文 ===\n" + "\n\n".join(parts) + "\n"


# 全局单例
memory_manager = MemoryManager()

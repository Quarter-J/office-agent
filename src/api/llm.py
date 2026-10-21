# -*- coding: utf-8 -*-
"""LLM model integration - DeepSeek API (compatible with OpenAI format)."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from src.config.settings import settings
from src.utils.execution import operation, remaining_seconds
from src.api.context import fit_payload, is_context_overflow, ModelOutputLimitError
from src.utils.logger import logger


class ChatDeepSeek(ChatOpenAI):
    """ChatOpenAI wrapper for DeepSeek reasoning model compatibility.

    DeepSeek reasoning models return `reasoning_content` in responses.
    The API requires this to be passed back in subsequent requests,
    but LangChain's _convert_dict_to_message (module-level function)
    discards it. We override _create_chat_result to capture it and
    _get_request_payload to inject it back into serialized messages.
    """

    def invoke(self, input, config=None, **kwargs):
        with operation("llm", settings.llm_timeout_seconds):
            for attempt in range(2):
                request = {**kwargs, "_context_scale": 0.5 if attempt else 1.0,
                           "timeout": remaining_seconds(settings.llm_timeout_seconds)}
                try:
                    response = super().invoke(input, config=config, **request)
                except Exception as exc:
                    if attempt == 0 and is_context_overflow(exc):
                        logger.info("[Context] 服务端拒绝上下文长度，收缩模型输入后重试一次")
                        continue
                    raise
                if response.response_metadata.get("finish_reason") == "length":
                    raise ModelOutputLimitError("模型输出达到长度上限，结果不完整；未继续执行后续动作。请缩小生成范围或调整输出预算。")
                return response

    def _create_chat_result(self, response, generation_info=None):
        """Capture reasoning_content from raw API response."""
        result = super()._create_chat_result(response, generation_info)

        response_dict = (
            response if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )

        for i, choice in enumerate(response_dict.get("choices", [])):
            msg_dict = choice.get("message", {})
            reasoning = msg_dict.get("reasoning_content")
            if reasoning and i < len(result.generations):
                result.generations[i].message.additional_kwargs["reasoning_content"] = reasoning

        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        """Inject reasoning_content back into serialized messages."""
        scale = kwargs.pop("_context_scale", 1.0)
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        messages = self._convert_input(input_).to_messages()
        api_messages = payload.get("messages", [])

        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and i < len(api_messages):
                reasoning = msg.additional_kwargs.get("reasoning_content")
                if reasoning:
                    api_messages[i]["reasoning_content"] = reasoning

        if "deepseek" in str(self.openai_api_base).lower() or "deepseek" in self.model_name.lower():
            payload["max_tokens"] = payload.pop("max_completion_tokens", settings.llm_output_tokens)
        payload, stats = fit_payload(payload, scale=scale)
        if stats["before"] != stats["after"]:
            logger.info(f"[Context] 模型输入估算 {stats['before']} -> {stats['after']}，预算 {stats['budget']}，"
                        f"移除 {stats['dropped_turns']} 个旧轮次")
        return payload


# ============================================================
# LLM Instance
# ============================================================
llm = ChatDeepSeek(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    model=settings.llm_model_name,
    temperature=settings.llm_temperature,
    timeout=settings.llm_timeout_seconds,
    max_retries=0,
)


def get_llm():
    """Get LLM instance."""
    if llm is None:
        raise ValueError("LLM not configured! Set OPENAI_API_KEY in .env")
    return llm

"""
短期记忆模块

基于 Redis 的会话级上下文存储，支持 TTL 自动过期。
Redis 不可用时自动降级为内存字典。
"""

import json
import time
import threading
import uuid
from typing import Optional
from src.utils.logger import logger


class _InMemoryFallback:
    """Redis 不可用时的内存回退实现"""

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.RLock()

    def _cleanup(self, key: str):
        if key in self._store and self._store[key]["expires"] < time.monotonic():
            del self._store[key]

    def lrange(self, key: str, start: int, end: int) -> list:
        with self._lock:
            self._cleanup(key)
            data = self._store.get(key, {}).get("value", [])
            stop = end + 1 if end >= 0 else (None if end == -1 else end + 1)
            return data[start:stop]

    def rpush(self, key: str, value: str):
        with self._lock:
            self._cleanup(key)
            if key not in self._store:
                self._store[key] = {"value": [], "expires": float("inf")}
            self._store[key]["value"].append(value)

    def ltrim(self, key: str, start: int, end: int):
        with self._lock:
            self._cleanup(key)
            if key in self._store:
                self._store[key]["value"] = self.lrange(key, start, end)

    def llen(self, key: str) -> int:
        with self._lock:
            self._cleanup(key)
            return len(self._store.get(key, {}).get("value", []))

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._cleanup(key)
            entry = self._store.get(key)
            return entry["value"] if entry else None

    def setex(self, key: str, ttl: int, value: str):
        with self._lock:
            self._store[key] = {"value": value, "expires": time.monotonic() + ttl}

    def expire(self, key: str, ttl: int):
        with self._lock:
            if key in self._store:
                self._store[key]["expires"] = time.monotonic() + ttl

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def commit_summary(self, turns_key, summary_key, expected, old_summary, summary, count, ttl):
        with self._lock:
            if self.lrange(turns_key, 0, len(expected) - 1) != expected or (self.get(summary_key) or "") != old_summary:
                return False
            self.setex(summary_key, ttl, summary)
            self.ltrim(turns_key, count, -1)
            self.expire(turns_key, ttl)
            return True

    def ping(self) -> bool:
        return True


class ShortTermMemory:
    """基于 Redis 的短期会话记忆"""

    def __init__(self, redis_url: str = "", ttl: int = 1800, max_turns: int = 20):
        self.ttl = ttl
        self.max_turns = max_turns
        self._client = self._connect(redis_url)

    def _connect(self, redis_url: str):
        if not redis_url:
            logger.info("[Memory] Redis URL 未配置，使用内存回退")
            return _InMemoryFallback()
        try:
            import redis
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info(f"[Memory] Redis 已连接: {redis_url}")
            return client
        except Exception as e:
            logger.warning(f"[Memory] Redis 连接失败 ({e})，使用内存回退")
            return _InMemoryFallback()

    def _turns_key(self, session_id: str) -> str:
        return f"office:session:{session_id}:turns"

    def _summary_key(self, session_id: str) -> str:
        return f"office:session:{session_id}:summary"

    def get_conversation(self, session_id: str) -> list[dict]:
        """获取最近的对话记录"""
        key = self._turns_key(session_id)
        raw_list = self._client.lrange(key, 0, -1)
        turns = []
        for raw in raw_list:
            try:
                turns.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return turns

    def add_turn(self, session_id: str, role: str, content: str):
        """添加一轮对话"""
        if not content:
            return
        key = self._turns_key(session_id)
        turn = json.dumps({"id": uuid.uuid4().hex, "role": role, "content": content}, ensure_ascii=False)
        self._client.rpush(key, turn)
        self._client.expire(key, self.ttl)

        # 超过上限时裁剪（保留最新的 max_turns 条）
        current_len = self._client.llen(key)
        if current_len > self.max_turns:
            self._client.ltrim(key, current_len - self.max_turns, -1)

    def get_summary(self, session_id: str) -> str:
        """获取会话摘要"""
        summary = self._client.get(self._summary_key(session_id))
        return summary or ""

    def update_summary(self, session_id: str, summary: str):
        """更新会话摘要"""
        key = self._summary_key(session_id)
        self._client.setex(key, self.ttl, summary)

    def get_turn_count(self, session_id: str) -> int:
        """获取消息条数；保留旧方法名以兼容调用方。"""
        return self._client.llen(self._turns_key(session_id))

    def commit_summary(self, session_id, expected_turns, old_summary, summary, count):
        """Atomically replace summary and remove exactly the summarized prefix.

        Concurrent appends survive. A changed/expired prefix or summary rejects
        the commit, so a slow summarizer cannot delete another request's data.
        """
        if not summary or not 0 < count < len(expected_turns):
            return False
        expected = [json.dumps(turn, ensure_ascii=False) for turn in expected_turns]
        turns_key, summary_key = self._turns_key(session_id), self._summary_key(session_id)
        if isinstance(self._client, _InMemoryFallback):
            return self._client.commit_summary(turns_key, summary_key, expected, old_summary, summary, count, self.ttl)
        script = """
            if (redis.call('GET', KEYS[2]) or '') ~= ARGV[1] then return 0 end
            local n = tonumber(ARGV[5])
            local items = redis.call('LRANGE', KEYS[1], 0, n - 1)
            if #items ~= n then return 0 end
            for i = 1, n do
                if items[i] ~= ARGV[5 + i] then return 0 end
            end
            redis.call('SETEX', KEYS[2], ARGV[3], ARGV[2])
            redis.call('LTRIM', KEYS[1], ARGV[4], -1)
            redis.call('EXPIRE', KEYS[1], ARGV[3])
            return 1
        """
        return bool(self._client.eval(script, 2, turns_key, summary_key,
                                      old_summary, summary, self.ttl, count, len(expected), *expected))

    def clear(self, session_id: str):
        """清空会话"""
        self._client.delete(self._turns_key(session_id))
        self._client.delete(self._summary_key(session_id))

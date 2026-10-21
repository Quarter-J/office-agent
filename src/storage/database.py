"""
数据库基础模块 - SQLite 本地持久化

提供任务记录、日志、用户配置的持久化存储能力。
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime


DB_PATH = Path("data/db/office_agent.db")


def get_connection() -> sqlite3.Connection:
    """获取数据库连接（自动创建库和表）"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection):
    """初始化基础表结构"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL,
            intent TEXT,
            status TEXT DEFAULT 'pending',
            input_data TEXT,
            output_data TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS message_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            receiver TEXT,
            content TEXT,
            status TEXT DEFAULT 'pending',
            retry_count INTEGER DEFAULT 0,
            error_message TEXT,
            scheduled_at TIMESTAMP,
            sent_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS report_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT,
            report_date TEXT NOT NULL,
            file_path TEXT,
            content_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS scheduler_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT NOT NULL,
            task_type TEXT NOT NULL,
            cron_expression TEXT,
            enabled INTEGER DEFAULT 1,
            config TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default',
            category TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            source_intent TEXT,
            confidence REAL DEFAULT 0.8,
            access_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, category, fact_key)
        );

        CREATE TABLE IF NOT EXISTS interaction_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default',
            intent TEXT,
            summary TEXT NOT NULL,
            input_snapshot TEXT,
            output_snapshot TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            label TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            attrs TEXT NOT NULL DEFAULT '{}',
            source_file TEXT,
            source_page INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_entity
            ON entities(entity_type, canonical_name, label);
    """)
    conn.commit()


def insert(table: str, data: dict) -> int:
    """插入记录"""
    conn = get_connection()
    try:
        keys = ", ".join(data.keys())
        values = ", ".join("?" for _ in data)
        sql = f"INSERT INTO {table} ({keys}) VALUES ({values})"
        cursor = conn.execute(sql, list(data.values()))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update(table: str, data: dict, where: str, params: tuple = ()):
    """更新记录"""
    conn = get_connection()
    try:
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        conn.execute(sql, list(data.values()) + list(params))
        conn.commit()
    finally:
        conn.close()


def upsert(table: str, data: dict, conflict_keys: list) -> int:
    """插入或更新记录（基于冲突键）"""
    conn = get_connection()
    try:
        keys = ", ".join(data.keys())
        values = ", ".join("?" for _ in data)
        update_clause = ", ".join(
            f"{k} = excluded.{k}" for k in data.keys() if k not in conflict_keys
        )
        conflict = ", ".join(conflict_keys)
        sql = f"INSERT INTO {table} ({keys}) VALUES ({values}) ON CONFLICT({conflict}) DO UPDATE SET {update_clause}"
        cursor = conn.execute(sql, list(data.values()))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def query(table: str, where: str = "1=1", params: tuple = (), order_by: str = "id DESC", limit: int = 100):
    """查询记录"""
    conn = get_connection()
    try:
        cursor = conn.execute(
            f"SELECT * FROM {table} WHERE {where} ORDER BY {order_by} LIMIT ?",
            params + (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

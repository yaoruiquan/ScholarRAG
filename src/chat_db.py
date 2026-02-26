"""
聊天历史数据库模块
使用 SQLite 持久化存储对话历史
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager


# 数据库路径
DB_PATH = Path("./data/chat_history.db")


@contextmanager
def get_db_connection():
    """获取数据库连接（上下文管理器）"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # 返回字典格式
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """初始化数据库表"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 会话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                kb_name TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # 消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        conn.commit()


def create_conversation(kb_name: str = "未知") -> int:
    """创建新会话，返回会话 ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO conversations (title, kb_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("新对话", kb_name, now, now)
        )
        conn.commit()
        return cursor.lastrowid or 0


def add_message(conversation_id: int, role: str, content: str):
    """添加消息到会话"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now)
        )
        
        # 更新会话标题（使用第一条用户消息）
        if role == "user":
            cursor.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            if row and row["title"] == "新对话":
                title = content[:30] + "..." if len(content) > 30 else content
                cursor.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, conversation_id)
                )
        
        conn.commit()


def get_messages(conversation_id: int) -> List[dict]:
    """获取会话的所有消息"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,)
        )
        return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]


def get_conversations(limit: int = 20) -> List[dict]:
    """获取最近的会话列表"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, title, kb_name, created_at, updated_at 
               FROM conversations 
               ORDER BY updated_at DESC 
               LIMIT ?""",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]


def delete_conversation(conversation_id: int):
    """删除会话及其消息"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


def rename_conversation(conversation_id: int, new_title: str):
    """重命名会话"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, now, conversation_id)
        )
        conn.commit()


def clear_all_history():
    """清空所有历史"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        conn.commit()


# 初始化数据库
init_db()

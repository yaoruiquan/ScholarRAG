"""
配置管理模块
集中管理应用配置
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """应用配置类"""
    
    # 千问 (Qwen) API 配置
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_API_BASE: str = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen-plus")
    
    # FAISS 向量数据库配置
    FAISS_PERSIST_DIR: str = os.getenv("FAISS_PERSIST_DIR", "./data/faiss_db")
    
    # Ollama Embedding 配置
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    # 文档处理配置
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5
    
    # BM25 + Vector 混合检索权重
    BM25_WEIGHT: float = 0.5
    VECTOR_WEIGHT: float = 0.5


config = Config()

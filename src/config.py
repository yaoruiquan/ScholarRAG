"""
配置管理模块
集中管理应用配置参数
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """应用配置类"""
    
    # ==================== 千问 (Qwen) API 配置 ====================
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_API_BASE: str = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen-plus")
    
    # ==================== Ollama 配置 ====================
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    
    # ==================== 数据目录配置 ====================
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    FAISS_PERSIST_DIR: str = os.getenv("FAISS_PERSIST_DIR", "./data/faiss_db")
    
    # ==================== 文档处理配置 ====================
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    # ==================== 检索配置 ====================
    # BM25 + Vector 混合检索权重
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    
    # 检索数量
    INITIAL_TOP_K: int = int(os.getenv("INITIAL_TOP_K", "30"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "15"))
    
    # ==================== 对话配置 ====================
    MAX_HISTORY_ROUNDS: int = 5  # 最大对话历史轮数


# 全局配置实例
config = Config()

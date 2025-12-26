# ScholarRAG

基于 RAG 技术的科研论文智能问答系统

## ✨ 功能特性

- 📚 **多格式文档支持**：PDF, Word, TXT, Markdown, PPT
- 🔍 **混合检索**：BM25 + 向量检索 + LLM Reranker
- 🧠 **Query Expansion**：多查询扩展提高召回率
- 💬 **对话记忆**：支持多轮对话上下文关联
- 💾 **历史持久化**：SQLite 存储对话历史
- 📊 **RAG 评估**：LLM-as-Judge 四维评估指标

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| Embedding | bge-m3 (Ollama) |
| Vector DB | FAISS |
| LLM | Qwen-plus (阿里云) |
| 前端 | Streamlit |
| 数据库 | SQLite |

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/yaoruiquan/ScholarRAG.git
cd ScholarRAG

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 .\venv\Scripts\Activate.ps1  # Windows

# 安装依赖
pip install -r requirements.txt
```

## ⚙️ 配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API Key：
```
QWEN_API_KEY=your_api_key_here
```

3. 确保 Ollama 运行并拉取 bge-m3 模型：
```bash
ollama pull bge-m3
```

## 🚀 运行

```bash
streamlit run app.py
```

访问 http://localhost:8501

## 📁 项目结构

```
ScholarRAG/
├── app.py                 # Streamlit 主应用
├── README.md
├── requirements.txt
├── .env.example           # 环境变量模板
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── rag_chain.py       # RAG 核心逻辑
│   ├── ingest.py          # 文档处理和向量化
│   ├── reranker.py        # LLM Reranker
│   ├── query_expansion.py # 查询扩展
│   ├── document_loader.py # 多格式文档加载
│   └── chat_db.py         # SQLite 聊天历史
│
├── evaluation/
│   ├── __init__.py
│   └── ragas_eval.py      # RAG 评估模块
│
└── data/                  # 知识库数据 (Git 忽略)
```

## 📊 评估指标

| 指标 | 说明 |
|------|------|
| 忠实度 (Faithfulness) | 回答是否基于上下文 |
| 相关性 (Relevance) | 回答是否切题 |
| 完整性 (Completeness) | 信息是否完整 |
| 连贯性 (Coherence) | 表达是否清晰 |

## 📝 许可证

MIT License

# ScholarRAG 项目开发总结

## 1. 项目概况
**ScholarRAG** 是一个基于 RAG (Retrieval-Augmented Generation) 技术的科研论文智能问答助手。项目旨在帮助科研人员高效地从 PDF 文论文中提取信息、通过对话方式获取答案，并提供精确的原文引用。

### 核心功能
*   **多源 PDF 处理**：支持上传多个 PDF 文档，自动进行清洗、切分。
*   **知识库管理**：支持创建多个独立知识库（如按课题分类），支持切换、加载和删除 knowledge base。
*   **混合检索系统**：结合关键字检索 (BM25) 和 语义向量检索 (Vector Search) 提高召回准确率。
*   **智能问答**：基于通义千问 (Qwen-Plus) 大模型生成回答，支持流式输出和对话历史记忆。
*   **可视化交互**：基于 Streamlit 构建的现代化 Web 界面，包含实时进度条、侧边栏配置等。

---

## 2. 技术架构演进

项目开发过程中，技术栈经历了多次迭代和优化，以适应本地环境（Windows）并解决兼容性问题。

| 组件 | 初始方案 | 最终方案 | 变更原因 |
| :--- | :--- | :--- | :--- |
| **LLM** | DeepSeek | **Qwen-Plus (通义千问)** | DeepSeek API 响应不稳定，Qwen 表现更佳且兼容 OpenAI 格式。 |
| **Embedding** | HuggingFace (本地) | **Ollama (nomic-embed-text)** | HuggingFace本地模型依赖 PyTorch，在 Windows 环境下导致 DLL 加载错误；Ollama 服务化部署更轻量稳定。 |
| **Vector DB** | ChromaDB | **FAISS** | ChromaDB 在 Windows 下与 Streamlit 多线程环境存在严重的进程锁/死锁问题，导致应用卡死；FAISS 更轻量且无此问题。 |
| **Integrations** | `langchain` (旧版) | **`langchain-classic` + `langchain-community`** | 适配 LangChain v0.2/v0.3 的架构调整，解决 `create_retrieval_chain` 等 API 迁移问题。 |
| **API 调用** | `langchain-ollama` SDK | **原生 `httpx` 请求** | `langchain_ollama` 官方包在本地环境频发 502 错误，改为自定义 `CustomOllamaEmbeddings` 类直接调用 HTTPs 接口。 |

---

## 3. 开发过程中的关键问题与解决方案

### 3.1 向量数据库死锁 (Blocker)
*   **问题现象**：在 Streamlit 中点击"建立知识库"后，进度条卡在 0% 或第一批次，后端无报错但进程挂起。
*   **根因分析**：ChromaDB (使用 SQLite/DuckDB 后端) 在 Windows 的多线程/多进程环境（Streamlit 运行机制）中容易发生数据库文件锁冲突，导致死锁。
*   **解决方案**：放弃 ChromaDB，迁移至 **FAISS (Facebook AI Similarity Search)**。FAISS 是纯内存索引+本地文件序列化，不依赖复杂的数据库锁机制，完美解决了卡死问题。

### 3.2 Ollama 502 错误
*   **问题现象**：使用 `OllamaEmbeddings` 调用本地模型时，频繁报 `502 Bad Gateway` 或连接错误，但 curl 测试正常。
*   **解决方案**：编写 `CustomOllamaEmbeddings` 类，继承 LangChain 的 `Embeddings` 接口，底层使用 `httpx`/`requests` 直接向 `http://localhost:11434/api/embeddings` 发送 HTTP 请求，绕过了可能有问题的封装层。

### 3.3 PyTorch DLL 加载失败
*   **问题现象**：`OSError: [WinError 1114] DLL initialization failed`。
*   **解决方案**：移除重量级的 `torch` 依赖，转而使用服务化的 Ollama 提供 Embedding 能力，降低了环境依赖复杂度。

### 3.4 LangChain 版本兼容性
*   **问题现象**：`ModuleNotFoundError: No module named 'langchain.chains'` 等导入错误。
*   **解决方案**：识别到 LangChain 核心功能拆分，引入 `langchain-classic` 用于向后兼容旧版 Chain 结构，或更新导入路径到 `langchain.chains.combine_documents` 等新位置。

---

## 4. 当前技术栈清单

*   **语言**: Python 3.12
*   **Web 框架**: Streamlit
*   **大模型**: Qwen-Plus (via DashScope/OpenAI Protocol)
*   **Embedding**: Ollama (nomic-embed-text)
*   **向量库**: FAISS (CPU version)
*   **编排框架**: LangChain (Core, Community, OpenAI, Text Splitters)
*   **PDF 处理**: PyMuPDF (fitz)
*   **检索算法**: BM25 + Vector Ensemble

## 5. 后续优化方向
1.  **引文跳转**：在前端实现点击引用 `[Page X]` 直接跳转到 PDF 对应页面的功能。
2.  **切分优化**：针对论文的双栏排本进行专门的布局分析优化。
3.  **模型微调**：在特定领域数据上微调 Embedding 模型以提升检索匹配度。

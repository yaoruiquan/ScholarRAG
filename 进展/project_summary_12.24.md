# ScholarRAG 项目进展总结 - 2024.12.24

## 一、项目进展

### 1. RAG 评估框架实现
- ✅ 创建了 `evaluation/ragas_eval.py` 评估模块
- ✅ 使用 LLM-as-Judge 方式实现四维评估指标：
  - **忠实度** (Faithfulness)：回答是否基于上下文
  - **相关性** (Relevance)：回答是否切题
  - **完整性** (Completeness)：信息是否完整
  - **连贯性** (Coherence)：表达是否清晰

### 2. Embedding 模型升级
- ✅ 从 `nomic-embed-text` (768d) 升级到 `bge-m3` (1024d)
- ✅ 支持更好的中英文多语言理解

### 3. 检索策略优化
- ✅ 调整 BM25:Vector 权重为 0.3:0.7（提高语义检索权重）
- ✅ 实现 Reranker 机制（LLM-based）
- ✅ 检索流程：`BM25+Vector (Top-30) → LLM Reranker → Top-15`

### 4. Prompt 优化
- ✅ 强化上下文约束，禁止模型编造信息
- ✅ 强制引用格式 `[文件名 Page X]`

### 5. PDF 表格提取
- ✅ 集成 `pdfplumber` 提取表格
- ✅ 将表格转换为结构化文本存入知识库

---

## 二、问题和解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 忠实度低 (2.0/5) | Prompt 约束不足 | 优化 Prompt，禁止外部知识 |
| 表格内容检索不全 | PDF 表格解析不完整 | 使用 pdfplumber 提取表格 |
| PyTorch DLL 加载失败 | Windows 兼容性问题 | 使用 Qwen API 直接调用替代本地模型 |
| Reranker 模型乱码 | Ollama reranker 模型不支持评分 | 改用 LLM Reranker (Qwen) |
| RerankingRetriever 不兼容 | 未实现 LangChain Runnable 接口 | 使用 RunnableLambda 包装 |

---

## 三、技术架构变化

### 当前架构
```
┌─────────────────────────────────────────────────────────┐
│                    ScholarRAG 系统                       │
├─────────────────────────────────────────────────────────┤
│  PDF → pdfplumber(表格) + PyMuPDF(文本) → 文档切分      │
│                         ↓                                │
│  bge-m3 Embedding (Ollama) → FAISS 向量数据库           │
│                         ↓                                │
│  BM25 + Vector 混合检索 (权重 0.3:0.7)                   │
│                         ↓                                │
│  LLM Reranker (Qwen API) → Top-15                        │
│                         ↓                                │
│  Qwen LLM 生成答案                                       │
└─────────────────────────────────────────────────────────┘
```

### 配置参数
| 参数 | 值 |
|------|------|
| Embedding 模型 | bge-m3 (1024d) |
| INITIAL_TOP_K | 30 |
| RERANK_TOP_K | 15 |
| BM25 权重 | 0.3 |
| Vector 权重 | 0.7 |
| LLM | Qwen-plus |

---

## 四、评估结果

### 最新评估 (2024.12.24 18:16)
| 指标 | 得分 |
|------|------|
| 忠实度 | 4.20/5 |
| 相关性 | 4.40/5 |
| 完整性 | 4.20/5 |
| 连贯性 | 4.80/5 |
| **综合** | **4.40/5** |

---

## 五、下一步优化方向

### P0 - 高优先级
1. **解决 Windows PyTorch 兼容问题** - 考虑使用 Docker 部署
2. **优化 LLM Reranker 效率** - 批量评分减少 API 调用次数

### P1 - 中优先级
3. **语义分块优化** - 使用 semantic chunking 替代固定大小切分
4. **Query 扩展** - 对用户问题进行同义词扩展提高召回

### P2 - 低优先级
5. **多模态支持** - 支持 PDF 中的图表理解
6. **缓存优化** - 对常见问题缓存答案

---

## 六、待解决 Bug

1. 新加载知识库时直接更新聊天记录
2. 构建知识库时显示具体加载块数进度
3. 知识库管理界面：新建完知识库后，旧库名称和文件仍显示

"""
RAG 链模块
实现 BM25 + Vector 混合检索与问答
使用 Ollama 本地 Embedding + 千问 LLM
"""

import os
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 加载环境变量
load_dotenv()


# ==================== 配置常量 ====================

FAISS_PERSIST_DIR = "./data/faiss_db"
COLLECTION_NAME = "scholar_rag"

# 检索权重配置（提高语义检索权重）
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7

# 检索配置
INITIAL_TOP_K = 30    # 初始检索数量
RERANK_TOP_K = 15     # 重排后保留数量（送入 LLM）
USE_RERANKER = True   # 启用 Ollama Reranker


# ==================== Prompt Template ====================

SYSTEM_PROMPT = """你是一个严谨的科研论文助手。请严格按照以下规则回答问题：

【核心规则 - 必须遵守】
1. **只使用提供的上下文**回答问题，禁止使用任何外部知识或常识
2. 如果上下文不包含相关信息，必须明确回答："根据提供的文档，无法找到相关信息"
3. 每个关键论点必须标注来源：[文件名, Page X]
4. 禁止编造、推测或添加上下文中没有的内容

【回答格式】
1. 首先给出 1-2 句简洁的结论
2. 然后详细解释，每段末尾标注引用来源
3. 使用专业的学术语言

【上下文内容】
{context}
"""

# 支持对话历史的用户提示
USER_PROMPT_WITH_HISTORY = """【对话历史】
{chat_history}

【当前问题】{input}

请严格基于上下文回答，标注每个论点的引用来源 [文件名, Page X]。如果上下文中没有相关信息，直接说明无法回答。
注意：你可以参考对话历史来理解用户问题的上下文（如代词指代），但回答必须基于上下文内容。"""

# 无对话历史的用户提示
USER_PROMPT = """【问题】{input}

请严格基于上下文回答，标注每个论点的引用来源 [文件名, Page X]。如果上下文中没有相关信息，直接说明无法回答。"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT)
])

# 带历史的 Prompt 模板
RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT_WITH_HISTORY)
])


# ==================== 核心函数 ====================

def get_embeddings():
    """
    获取 Embedding 模型
    
    使用 bge-m3 多语言模型（中英文效果都很好）
    """
    from src.ingest import CustomOllamaEmbeddings
    embeddings = CustomOllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    return embeddings


def load_vectorstore(
    persist_directory: str = FAISS_PERSIST_DIR
) -> FAISS:
    """
    加载已持久化的 FAISS 向量数据库
    
    Args:
        persist_directory: 持久化目录
        
    Returns:
        FAISS 向量数据库实例
    """
    embeddings = get_embeddings()
    
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore


def create_ensemble_retriever(
    vectorstore: FAISS,
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
    top_k: int = INITIAL_TOP_K
) -> EnsembleRetriever:
    """
    创建 BM25 + Vector 混合检索器 (EnsembleRetriever)
    
    Args:
        vectorstore: Chroma 向量数据库
        bm25_weight: BM25 检索器权重
        vector_weight: 向量检索器权重
        top_k: 返回的文档数量
        
    Returns:
        EnsembleRetriever 混合检索器
    """
    print("  🔄 正在获取文档用于 BM25...")
    
    # 1. 从 FAISS 获取所有文档用于 BM25
    try:
        # FAISS 使用 docstore 存储文档
        docstore = vectorstore.docstore
        documents = []
        
        for doc_id in vectorstore.index_to_docstore_id.values():
            doc = docstore.search(doc_id)
            if doc and hasattr(doc, 'page_content'):
                documents.append(doc)
        
        print(f"  📄 获取到 {len(documents)} 个文档")
        
        if not documents:
            raise ValueError("知识库中没有文档")
        
        print(f"  📚 创建 BM25 检索器 ({len(documents)} 个文档)...")
        
        # 2. 创建 BM25 检索器
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = top_k
        
    except Exception as e:
        print(f"  ⚠️ BM25 创建失败: {e}，使用纯向量检索")
        # 如果 BM25 失败，只使用向量检索
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        return vector_retriever  # type: ignore
    
    print("  🔍 创建向量检索器...")
    
    # 3. 创建向量检索器
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    print("  🔗 创建混合检索器...")
    
    # 4. 创建混合检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )
    
    return ensemble_retriever


class RerankingRetriever:
    """
    带 Reranker 的检索器包装类
    实现 LangChain Retriever 接口
    """
    
    def __init__(self, base_retriever: EnsembleRetriever, rerank_top_k: int = RERANK_TOP_K):
        self.base_retriever = base_retriever
        self.rerank_top_k = rerank_top_k
        self._reranker = None
    
    @property
    def reranker(self):
        """延迟加载 Reranker"""
        if self._reranker is None and USE_RERANKER:
            from src.reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker
    
    def invoke(self, query: str, config=None) -> List[Document]:
        """检索并重排序"""
        # 处理不同类型的输入
        if isinstance(query, dict):
            query = query.get("input", str(query))
        
        # 1. 初始检索
        docs = self.base_retriever.invoke(query)
        
        # 2. Reranker 重排序
        if self.reranker and docs:
            docs = self.reranker.rerank(query, docs, self.rerank_top_k)
        
        return docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """兼容旧版 LangChain 接口"""
        return self.invoke(query)
    
    def with_config(self, **kwargs):
        """兼容 LangChain Runnable 接口"""
        return self


def create_reranking_retriever(vectorstore: FAISS):
    """
    创建带 Query Expansion 和 Reranker 的检索器
    
    Args:
        vectorstore: FAISS 向量数据库
        
    Returns:
        兼容 LangChain 的检索器
    """
    from langchain_core.runnables import RunnableLambda
    from src.query_expansion import expand_query
    
    # 创建基础混合检索器
    base_retriever = create_ensemble_retriever(vectorstore)
    
    # 创建 Reranker 包装器
    reranker_wrapper = RerankingRetriever(base_retriever)
    
    # 使用 RunnableLambda 包装以兼容 LangChain
    def retrieve_with_expansion_and_rerank(query):
        if isinstance(query, dict):
            query = query.get("input", str(query))
        
        # 1. Query Expansion - 生成多个查询变体
        expanded_queries = expand_query(query, num_variants=3)
        
        # 2. 对每个查询变体进行检索
        all_docs = []
        seen_contents = set()
        
        for q in expanded_queries:
            docs = base_retriever.invoke(q)
            for doc in docs:
                # 去重（基于内容）
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        print(f"  📚 Multi-Query 检索: {len(expanded_queries)} 个查询 → {len(all_docs)} 个文档（去重后）")
        
        # 3. Reranker 重排序
        if USE_RERANKER and reranker_wrapper.reranker and all_docs:
            all_docs = reranker_wrapper.reranker.rerank(query, all_docs, RERANK_TOP_K)
        
        return all_docs
    
    return RunnableLambda(retrieve_with_expansion_and_rerank)


def get_llm() -> ChatOpenAI:
    """
    初始化千问 (Qwen) LLM
    
    通过 OpenAI 兼容接口连接阿里云 DashScope API
    
    Returns:
        ChatOpenAI 实例
    """
    from pydantic import SecretStr
    
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("请设置 QWEN_API_KEY 环境变量")
    
    api_base = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url=api_base,
        temperature=0.3,
    )
    
    return llm


def build_rag_chain(retriever, use_history: bool = False):
    """
    构建 RAG Chain
    
    Args:
        retriever: 检索器
        use_history: 是否使用对话历史
        
    Returns:
        RAG Chain
    """
    llm = get_llm()
    
    # 选择 Prompt 模板
    prompt = RAG_PROMPT_WITH_HISTORY if use_history else RAG_PROMPT
    
    # 创建文档处理链
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    
    # 创建完整 RAG 链
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )
    
    return rag_chain


# ==================== 对外接口 ====================

class ScholarRAG:
    """ScholarRAG 问答系统封装类"""
    
    def __init__(
        self,
        persist_directory: str = FAISS_PERSIST_DIR
    ):
        """
        初始化 ScholarRAG 系统
        
        Args:
            persist_directory: FAISS 持久化目录
        """
        print("🔄 正在初始化 ScholarRAG 系统...")
        
        # 加载向量数据库
        self.vectorstore = load_vectorstore(persist_directory)
        print("  ✅ 向量数据库加载完成")
        
        # 创建带 Reranker 的混合检索器
        self.retriever = create_reranking_retriever(self.vectorstore)
        if USE_RERANKER:
            print("  ✅ 混合检索器创建完成 (BM25 + Vector + Reranker)")
        else:
            print("  ✅ 混合检索器创建完成 (BM25 + Vector)")
        
        # 构建 RAG Chain
        self.rag_chain = build_rag_chain(self.retriever)
        print("  ✅ RAG Chain 构建完成")
        
        print("🎉 ScholarRAG 初始化完成！")
    
    def get_answer(self, question: str, chat_history: Optional[List[dict]] = None) -> Tuple[str, List[Document]]:
        """
        获取问题的答案（支持对话历史）
        
        Args:
            question: 用户问题
            chat_history: 对话历史列表 [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            (answer, source_documents) 元组
            - answer: 生成的答案
            - source_documents: 引用的源文档列表
        """
        print(f"\n📝 用户问题: {question}")
        
        # 构建输入
        invoke_input = {"input": question}
        
        # 如果有对话历史，格式化并添加
        if chat_history and len(chat_history) > 0:
            # 只保留最近 5 轮对话
            recent_history = chat_history[-10:]  # 最多 10 条消息（5轮）
            
            # 格式化对话历史
            history_text = ""
            for msg in recent_history:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")[:200]  # 截断过长内容
                history_text += f"{role}: {content}\n"
            
            invoke_input["chat_history"] = history_text.strip()
            print(f"💭 使用对话历史: {len(recent_history)} 条消息")
            
            # 使用带历史的 RAG Chain
            rag_chain = build_rag_chain(self.retriever, use_history=True)
            result = rag_chain.invoke(invoke_input)
        else:
            # 无历史，使用普通 RAG Chain
            result = self.rag_chain.invoke(invoke_input)
        
        answer = result.get("answer", "")
        source_docs = result.get("context", [])
        
        print(f"📚 检索到 {len(source_docs)} 个相关文档")
        if source_docs:
            for i, doc in enumerate(source_docs[:3]):
                print(f"  [{i+1}] {doc.page_content[:100]}...")
        print(f"💬 回答长度: {len(answer)} 字符")
        
        return answer, source_docs
    
    def format_sources(self, source_docs: List[Document]) -> str:
        """
        格式化源文档信息
        
        Args:
            source_docs: 源文档列表
            
        Returns:
            格式化后的引用信息字符串
        """
        if not source_docs:
            return "无引用文档"
        
        sources = []
        seen = set()
        
        for doc in source_docs:
            metadata = doc.metadata
            source = metadata.get("source", "未知来源")
            page = metadata.get("page", "?")
            
            # 提取文件名
            if "/" in source or "\\" in source:
                source = source.replace("\\", "/").split("/")[-1]
            
            key = f"{source}_p{page}"
            if key not in seen:
                seen.add(key)
                sources.append(f"- {source} (Page {page})")
        
        return "\n".join(sources)


def get_answer(question: str) -> Tuple[str, List[Document]]:
    """
    便捷函数：获取问题的答案
    
    Args:
        question: 用户问题
        
    Returns:
        (answer, source_documents) 元组
    """
    rag = ScholarRAG()
    return rag.get_answer(question)


# ==================== 测试入口 ====================

if __name__ == "__main__":
    # 测试问答
    rag = ScholarRAG()
    
    test_question = "请总结这篇论文的主要贡献"
    print(f"\n📝 测试问题: {test_question}")
    print("-" * 50)
    
    answer, sources = rag.get_answer(test_question)
    
    print(f"💡 回答:\n{answer}")
    print("-" * 50)
    print(f"📚 引用来源:\n{rag.format_sources(sources)}")

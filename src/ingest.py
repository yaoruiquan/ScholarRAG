"""
数据处理模块 - PDF 文档加载、清洗、切分与向量化存储
使用 Ollama 本地 Embedding 模型（nomic-embed-text）
"""

import os
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()


def load_pdfs(data_dir: str = "./data/pdfs") -> List[Document]:
    """
    加载指定目录下所有 PDF 文件（直接使用 PyMuPDF）
    
    Args:
        data_dir: PDF 文件所在目录
        
    Returns:
        Document 列表
    """
    documents = []
    pdf_dir = Path(data_dir)
    
    if not pdf_dir.exists():
        print(f"⚠️ 目录不存在: {data_dir}")
        return documents
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ 目录 {data_dir} 中没有找到 PDF 文件")
        return documents
    
    print(f"📄 找到 {len(pdf_files)} 个 PDF 文件")
    
    for pdf_path in pdf_files:
        try:
            print(f"  ⏳ 正在加载: {pdf_path.name}")
            
            # 使用 pdfplumber 提取表格
            import pdfplumber
            tables_text = []
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # 提取表格
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                # 将表格转换为结构化文本
                                table_str = format_table_as_text(table, page_num + 1)
                                if table_str:
                                    tables_text.append((page_num + 1, table_str))
            except Exception as e:
                print(f"    ⚠️ 表格提取失败: {e}")
            
            # 使用 PyMuPDF 提取普通文本
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text and str(text).strip():
                    documents.append(Document(
                        page_content=str(text),
                        metadata={
                            "source": str(pdf_path),
                            "page": page_num + 1
                        }
                    ))
            
            # 添加提取的表格作为单独的文档
            for page_num, table_text in tables_text:
                documents.append(Document(
                    page_content=table_text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_num,
                        "type": "table"
                    }
                ))
            
            page_count = len(doc)
            doc.close()
            print(f"  ✅ 加载完成: {pdf_path.name} ({page_count} 页, {len(tables_text)} 个表格)")
        except Exception as e:
            print(f"  ❌ 加载失败: {pdf_path.name} - {e}")
    
    return documents


def format_table_as_text(table: list, page_num: int) -> str:
    """
    将表格转换为结构化的自然语言描述
    """
    if not table or len(table) < 2:
        return ""
    
    # 获取表头
    headers = table[0]
    if not headers:
        return ""
    
    # 清理空值
    headers = [str(h).strip() if h else f"列{i+1}" for i, h in enumerate(headers)]
    
    rows_text = []
    for row in table[1:]:
        if not row:
            continue
        # 将每行转换为 "字段: 值" 格式
        row_parts = []
        for i, cell in enumerate(row):
            if cell and str(cell).strip():
                header = headers[i] if i < len(headers) else f"列{i+1}"
                row_parts.append(f"{header}: {str(cell).strip()}")
        if row_parts:
            rows_text.append("；".join(row_parts))
    
    if not rows_text:
        return ""
    
    return f"[表格内容 - 第{page_num}页]\n" + "\n".join(rows_text)


def clean_text(text: str) -> str:
    """
    文本清洗预处理
    
    1. 修复换行符造成的单词断裂 (hyphenation fix)
    2. 去除 References 之后的内容
    3. 规范化空白字符
    
    Args:
        text: 原始文本
        
    Returns:
        清洗后的文本
    """
    # 1. 修复连字符断词 (hyphenation fix)
    # 例如: "knowl-\nedge" -> "knowledge"
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 2. 去除 References/Bibliography 之后的内容
    # 匹配多种参考文献标题格式
    references_pattern = r'\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY|参考文献)\s*\n'
    match = re.search(references_pattern, text)
    if match:
        text = text[:match.start()]
    
    # 3. 将多个连续换行符替换为单个换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 4. 去除行首行尾多余空白
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # 5. 去除多余空格
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """
    对文档列表进行预处理
    
    Args:
        documents: 原始文档列表
        
    Returns:
        清洗后的文档列表
    """
    cleaned_docs = []
    
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content:  # 只保留非空文档
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            cleaned_docs.append(cleaned_doc)
    
    print(f"🧹 文档清洗完成: {len(documents)} -> {len(cleaned_docs)} 个有效文档")
    return cleaned_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 切分文档
    
    Args:
        documents: 文档列表
        chunk_size: 切片大小
        chunk_overlap: 切片重叠
        
    Returns:
        切分后的文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 文档切分完成: {len(documents)} 个文档 -> {len(chunks)} 个 chunks")
    
    return chunks


class CustomOllamaEmbeddings(Embeddings):
    """
    自定义 Ollama Embeddings 类
    继承 LangChain Embeddings 接口，直接使用 httpx 调用 API
    """
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.embed_url = f"{self.base_url}/api/embeddings"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        import httpx
        embeddings = []
        
        with httpx.Client(timeout=120.0) as client:
            for idx, text in enumerate(texts):
                try:
                    response = client.post(
                        self.embed_url,
                        json={"model": self.model, "prompt": text}
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings.append(data["embedding"])
                except Exception as e:
                    print(f"    ❌ Embedding #{idx} failed: {e}")
                    raise
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]


def get_embeddings():
    """
    获取 Embedding 模型
    
    使用 bge-m3 多语言模型（中英文效果都很好）
    
    Returns:
        Embeddings 实例
    """
    print("🔄 正在连接 Ollama Embedding 模型 (bge-m3)...")
    
    embeddings = CustomOllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    
    # 测试连接
    try:
        test_result = embeddings.embed_query("test")
        print(f"✅ Ollama Embedding 连接成功 (向量维度: {len(test_result)})")
    except Exception as e:
        print(f"⚠️ Ollama 连接测试失败: {e}")
    
    return embeddings


def create_vectorstore(
    chunks: List[Document],
    persist_directory: str = "./data/faiss_db",
    batch_size: int = 20,
    progress_callback=None
) -> "FAISS | None":
    """
    创建并持久化 FAISS 向量数据库（分批处理）
    
    Args:
        chunks: 文档切片列表
        persist_directory: 持久化存储目录
        batch_size: 每批处理的文档数量
        progress_callback: 进度回调函数 callback(current, total, message)
        
    Returns:
        Chroma 向量数据库实例
    """
    import shutil
    
    # 确保目录存在（清理旧数据）
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)
    
    # 获取 Embedding 模型
    embeddings = get_embeddings()
    
    total_chunks = len(chunks)
    print(f"💾 正在创建向量数据库，共 {total_chunks} 个 chunks（每批 {batch_size} 个）...")
    
    if progress_callback:
        progress_callback(0, total_chunks, "开始处理...")
    
    vectorstore = None
    
    # 分批处理
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"  📦 处理批次 {batch_num}/{total_batches} ({len(batch)} 个文档)...")
        
        if progress_callback:
            progress_callback(i, total_chunks, f"正在 Embedding 第 {batch_num}/{total_batches} 批...")
        
        if vectorstore is None:
            # 第一批：创建新的向量数据库
            vectorstore = FAISS.from_documents(
                documents=batch,
                embedding=embeddings
            )
        else:
            # 后续批次：添加到现有数据库
            vectorstore.add_documents(batch)
        
        print(f"  ✅ 批次 {batch_num} 完成")
    
    if progress_callback:
        progress_callback(total_chunks, total_chunks, "向量数据库创建完成！")
    
    # 保存 FAISS 索引
    if vectorstore:
        # 转为绝对路径并确保目录存在
        abs_path = Path(persist_directory).resolve()
        abs_path.mkdir(parents=True, exist_ok=True)
        print(f"  💾 保存到: {abs_path}")
        vectorstore.save_local(str(abs_path))
    
    print(f"✅ 向量数据库创建完成，已持久化到: {persist_directory}")
    
    return vectorstore


def load_existing_vectorstore(persist_directory: str = "./data/faiss_db") -> FAISS:
    """
    加载已存在的向量数据库
    
    Args:
        persist_directory: 持久化存储目录
        
    Returns:
        FAISS 向量数据库实例
    """
    embeddings = get_embeddings()
    
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"✅ 已加载现有向量数据库: {persist_directory}")
    return vectorstore


def ingest_pdfs(
    data_dir: str = "./data/pdfs",
    persist_directory: str = "./data/faiss_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> "FAISS | None":
    """
    完整的 PDF 数据摄入流程
    
    1. 加载 PDF 文件
    2. 清洗文本
    3. 切分文档
    4. 创建向量数据库
    
    Args:
        data_dir: PDF 文件目录
        persist_directory: 向量数据库存储目录
        chunk_size: 切片大小
        chunk_overlap: 切片重叠
        
    Returns:
        Chroma 向量数据库实例
    """
    print("=" * 50)
    print("🚀 开始 PDF 数据摄入流程")
    print("=" * 50)
    
    # Step 1: 加载 PDF
    documents = load_pdfs(data_dir)
    if not documents:
        raise ValueError("没有加载到任何文档，请检查 PDF 文件路径")
    
    # Step 2: 清洗文本
    cleaned_docs = preprocess_documents(documents)
    
    # Step 3: 切分文档
    chunks = split_documents(cleaned_docs, chunk_size, chunk_overlap)
    
    # Step 4: 创建向量数据库
    vectorstore = create_vectorstore(chunks, persist_directory)
    
    print("=" * 50)
    print(f"🎉 数据摄入完成！")
    print(f"   📄 处理文档: {len(documents)} 个")
    print(f"   ✂️ 生成切片: {len(chunks)} 个 chunks")
    print(f"   💾 存储位置: {persist_directory}")
    print("=" * 50)
    
    return vectorstore


# 命令行入口
if __name__ == "__main__":
    ingest_pdfs()

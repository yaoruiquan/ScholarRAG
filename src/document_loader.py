"""
统一文档加载器
支持多种文件格式：PDF, DOCX, TXT, MD, PPTX
增强元数据：章节检测、元素类型、字符统计
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document


# ==================== Metadata 增强函数 ====================

# 章节模式匹配（按优先级排序）
SECTION_PATTERNS = [
    (r"\babstract\b", "Abstract"),
    (r"\bintroduction\b", "Introduction"),
    (r"\brelated\s*work\b", "Related Work"),
    (r"\bbackground\b", "Background"),
    (r"\b(method|methodology)\b", "Methods"),
    (r"\b(experiment|experimental)\b", "Experiments"),
    (r"\bresult\b", "Results"),
    (r"\bdiscussion\b", "Discussion"),
    (r"\bconclusion\b", "Conclusion"),
    (r"\backnowledg", "Acknowledgments"),
    (r"\breference\b", "References"),
    (r"\bappendix\b", "Appendix"),
]


def detect_section(text: str, current_section: str = "Unknown") -> str:
    """
    检测文本所属章节
    
    Args:
        text: 文本内容
        current_section: 当前章节（用于延续）
    
    Returns:
        章节名称
    """
    # 取前 200 字符检测章节标题
    header = text[:200].lower()
    
    for pattern, section_name in SECTION_PATTERNS:
        if re.search(pattern, header, re.IGNORECASE):
            return section_name
    
    return current_section


def detect_element_type(text: str) -> str:
    """
    检测文本元素类型
    
    Returns:
        "text" | "table" | "figure_caption" | "equation"
    """
    text_lower = text.strip().lower()
    
    # 表格标记
    if text.startswith("[表格]") or text.startswith("[Table]"):
        return "table"
    
    # 图说明
    if re.match(r"^(figure|fig\.|图)\s*\d+", text_lower):
        return "figure_caption"
    
    # 公式（简单检测）
    if re.match(r"^(equation|eq\.|公式)\s*\d+", text_lower):
        return "equation"
    
    return "text"


# ==================== 文档加载函数 ====================

def load_pdf(file_path: str) -> List[Document]:
    """加载 PDF 文件（增强元数据）"""
    import fitz  # PyMuPDF
    
    documents = []
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        current_section = "Unknown"
        
        for page_num, page in enumerate(doc):  # type: ignore
            text = page.get_text()
            if text.strip():
                # 检测章节（会更新 current_section）
                current_section = detect_section(text, current_section)
                
                # 检测元素类型
                element_type = detect_element_type(text)
                
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "page": page_num + 1,
                        "file_type": "pdf",
                        # 增强元数据
                        "section": current_section,
                        "element_type": element_type,
                        "total_pages": total_pages,
                        "char_count": len(text),
                    }
                ))
        doc.close()
    except Exception as e:
        print(f"  ⚠️ PDF 加载失败 {file_path}: {e}")
    
    return documents


def load_docx(file_path: str) -> List[Document]:
    """加载 Word 文档（增强元数据）"""
    from docx import Document as DocxDocument
    
    documents = []
    try:
        doc = DocxDocument(file_path)
        
        # 提取所有段落文本
        full_text = []
        has_table = False
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        # 提取表格内容
        for table in doc.tables:
            has_table = True
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                if row_text:
                    full_text.append(f"[表格] {row_text}")
        
        if full_text:
            content = "\n".join(full_text)
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": 1,
                    "file_type": "docx",
                    # 增强元数据
                    "section": detect_section(content),
                    "element_type": "table" if has_table else "text",
                    "total_pages": 1,
                    "char_count": len(content),
                }
            ))
    except Exception as e:
        print(f"  ⚠️ DOCX 加载失败 {file_path}: {e}")
    
    return documents


def load_txt(file_path: str) -> List[Document]:
    """加载纯文本文件（增强元数据）"""
    documents = []
    try:
        # 尝试多种编码
        content = ""
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"  ⚠️ 无法解码文件 {file_path}")
            return []
        
        if content.strip():
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": 1,
                    "file_type": "txt",
                    # 增强元数据
                    "section": detect_section(content),
                    "element_type": "text",
                    "total_pages": 1,
                    "char_count": len(content),
                }
            ))
    except Exception as e:
        print(f"  ⚠️ TXT 加载失败 {file_path}: {e}")
    
    return documents


def load_markdown(file_path: str) -> List[Document]:
    """加载 Markdown 文件"""
    return load_txt(file_path)  # 与 TXT 处理相同


def load_pptx(file_path: str) -> List[Document]:
    """加载 PowerPoint 文件（增强元数据）"""
    from pptx import Presentation
    
    documents = []
    try:
        prs = Presentation(file_path)
        total_slides = len(prs.slides)
        
        for slide_num, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text = getattr(shape, "text", "")
                    if text.strip():
                        slide_text.append(text)
            
            if slide_text:
                content = "\n".join(slide_text)
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "page": slide_num + 1,
                        "file_type": "pptx",
                        # 增强元数据
                        "section": f"Slide {slide_num + 1}",
                        "element_type": "text",
                        "total_pages": total_slides,
                        "char_count": len(content),
                    }
                ))
    except Exception as e:
        print(f"  ⚠️ PPTX 加载失败 {file_path}: {e}")
    
    return documents


# 支持的文件格式映射
FILE_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".doc": load_docx,  # 尝试用 docx 加载
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pptx": load_pptx,
    ".ppt": load_pptx,  # 尝试用 pptx 加载
}

# 支持的文件扩展名列表
SUPPORTED_EXTENSIONS = list(FILE_LOADERS.keys())


def load_document(file_path: str) -> List[Document]:
    """
    加载单个文档
    
    Args:
        file_path: 文件路径
        
    Returns:
        Document 列表
    """
    ext = Path(file_path).suffix.lower()
    
    if ext not in FILE_LOADERS:
        print(f"  ⚠️ 不支持的文件格式: {ext}")
        return []
    
    loader = FILE_LOADERS[ext]
    return loader(file_path)


def load_documents_from_directory(
    directory: str, 
    extensions: Optional[List[str]] = None
) -> List[Document]:
    """
    从目录加载所有支持的文档
    
    Args:
        directory: 目录路径
        extensions: 指定要加载的扩展名列表，默认加载所有支持的格式
        
    Returns:
        Document 列表
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS
    
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"⚠️ 目录不存在: {directory}")
        return []
    
    # 遍历目录中的所有文件
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            print(f"  📄 加载: {file_path.name}")
            docs = load_document(str(file_path))
            documents.extend(docs)
    
    print(f"✅ 共加载 {len(documents)} 个文档片段")
    return documents


def get_supported_extensions_str() -> str:
    """获取支持的文件扩展名字符串（用于 UI 显示）"""
    return ", ".join(SUPPORTED_EXTENSIONS)

"""
ScholarRAG - 科研问答助手
Streamlit 应用入口
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
import re
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载环境变量
load_dotenv()


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="ScholarRAG - 科研问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 自定义样式 ====================

st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* 聊天消息样式 */
    .stChatMessage {
        border-radius: 12px;
    }
    
    /* 侧边栏标题 */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4F46E5;
        margin-bottom: 0.5rem;
    }
    
    /* 来源卡片样式 */
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #4F46E5;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* 进度提示 */
    .processing-hint {
        color: #6B7280;
        font-size: 0.9rem;
    }
    
    /* 侧边栏布局优化 */
    section[data-testid="stSidebar"] {
        width: 320px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* 侧边栏各板块间距优化 */
    .stExpander {
        margin-bottom: 0.5rem;
    }
    
    /* 减小侧边栏字体 */
    section[data-testid="stSidebar"] .stMarkdown {
        font-size: 0.9rem;
    }
    
    /* 按钮紧凑样式 */
    section[data-testid="stSidebar"] button {
        padding: 0.3rem 0.5rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Session State 初始化 ====================

def init_session_state():
    """初始化会话状态"""
    from src.chat_db import get_conversations, get_messages, create_conversation
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    if "knowledge_base_ready" not in st.session_state:
        st.session_state.knowledge_base_ready = False
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("QWEN_API_KEY", "")
    
    # 聊天历史管理（从数据库加载）
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    # 多知识库选择
    if "selected_kbs" not in st.session_state:
        st.session_state.selected_kbs = []  # 选中的知识库列表


init_session_state()


def save_api_key_to_env(api_key: str):
    """保存 API Key 到 .env 文件"""
    env_path = Path(__file__).parent / ".env"
    
    # 读取现有内容
    existing_lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
    
    # 更新或添加 QWEN_API_KEY
    found = False
    new_lines = []
    for line in existing_lines:
        if line.startswith("QWEN_API_KEY="):
            if api_key:
                new_lines.append(f"QWEN_API_KEY={api_key}\n")
            found = True
        else:
            new_lines.append(line)
    
    if not found and api_key:
        new_lines.append(f"QWEN_API_KEY={api_key}\n")
    
    # 写入文件
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# ==================== 侧边栏 ====================

# ==================== 侧边栏 ====================

import hashlib

def safe_kb_dirname(kb_name: str) -> str:
    """将知识库名称转换为安全的目录名（ASCII only）"""
    # 如果全是 ASCII，直接返回
    if kb_name.isascii():
        return kb_name
    # 否则用 hash 前缀 + 原名的 MD5
    return f"kb_{hashlib.md5(kb_name.encode('utf-8')).hexdigest()[:12]}"

def get_kb_name_mapping() -> dict:
    """获取目录名到显示名的映射"""
    mapping_file = Path("./data/.kb_names.json")
    if mapping_file.exists():
        import json
        with open(mapping_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_kb_name_mapping(dirname: str, display_name: str):
    """保存目录名到显示名的映射"""
    import json
    mapping_file = Path("./data/.kb_names.json")
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    
    mapping = get_kb_name_mapping()
    mapping[dirname] = display_name
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def save_kb_metadata(dirname: str, files: list, chunk_count: int):
    """保存知识库元数据（文件列表、块数等）"""
    import json
    metadata_file = Path(f"./data/{dirname}_db/.metadata.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "files": files,
        "chunk_count": chunk_count,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def get_kb_metadata(dirname: str) -> dict:
    """获取知识库元数据"""
    import json
    metadata_file = Path(f"./data/{dirname}_db/.metadata.json")
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": [], "chunk_count": 0}

def get_knowledge_bases():
    """获取所有知识库（返回显示名列表）"""
    data_dir = Path("./data")
    if not data_dir.exists():
        return []
    
    mapping = get_kb_name_mapping()
    
    # 查找所有以 _db 结尾的目录
    result = []
    for d in data_dir.iterdir():
        if d.is_dir() and d.name.endswith("_db"):
            dirname = d.name.replace("_db", "")
            # 优先使用映射的显示名
            display_name = mapping.get(dirname, dirname)
            result.append((dirname, display_name))
    
    return sorted(result, key=lambda x: x[1])

def delete_knowledge_base(kb_name):
    """删除知识库"""
    import shutil
    db_path = Path(f"./data/{kb_name}_db")
    if db_path.exists():
        try:
            shutil.rmtree(db_path)
            st.toast(f"✅ 已删除 knowledge base: {kb_name}")
            return True
        except Exception as e:
            st.error(f"❌ 删除失败: {e}")
            return False
    return False

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown('<p class="sidebar-title">⚙️ 设置</p>', unsafe_allow_html=True)
        
        # API Key 设置
        with st.expander("🔑 API 密钥设置", expanded=not st.session_state.api_key):
            api_key_input = st.text_input(
                "千问 API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="输入你的千问 API Key",
                key="api_key_input"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 保存", use_container_width=True):
                    if api_key_input:
                        st.session_state.api_key = api_key_input
                        os.environ["QWEN_API_KEY"] = api_key_input
                        save_api_key_to_env(api_key_input)
                        st.session_state.rag_system = None
                        st.success("✅ 已保存")
                        st.rerun()
                    else:
                        st.error("请输入 API Key")
            
            with col2:
                if st.button("🗑️ 清除", use_container_width=True):
                    st.session_state.api_key = ""
                    os.environ["QWEN_API_KEY"] = ""
                    save_api_key_to_env("")
                    st.rerun()
        
        # 知识库管理
        st.markdown("---")
        st.markdown("**📚 知识库管理**")
        
        # 1. 新建知识库
        with st.expander("➕ 新建知识库", expanded=False):
            new_kb_name = st.text_input("知识库名称", placeholder="例如: paper_v1")
            
            uploaded_files = st.file_uploader(
                "上传文档",
                type=["pdf", "docx", "doc", "txt", "md", "pptx", "ppt"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="支持 PDF, Word, TXT, Markdown, PPT"
            )
            
            if st.button("🚀 开始构建", use_container_width=True, type="primary"):
                if not new_kb_name:
                    st.error("请输入知识库名称")
                elif not uploaded_files:
                    st.error("请上传文档文件")
                elif not st.session_state.api_key:
                    st.error("请先设置 API Key")
                else:
                    # 检查名称是否合法
                    if not re.match(r'^[a-zA-Z0-9_\u4e00-\u9fa5]+$', new_kb_name):
                        st.error("名称仅支持中文、字母、数字和下划线")
                    else:
                        build_knowledge_base(uploaded_files, new_kb_name)
        
        # 2. 现有知识库列表
        kb_list = get_knowledge_bases()  # 返回 [(dirname, display_name), ...]
        
        if kb_list:
            # 显示名称列表
            display_names = [item[1] for item in kb_list]
            dir_names = [item[0] for item in kb_list]
            
            # 知识库选择（多选模式）
            selected_displays = st.multiselect(
                "选择知识库（可多选）",
                options=display_names,
                default=[st.session_state.get('current_kb')] if st.session_state.get('current_kb') in display_names else [],
                key="kb_multiselect"
            )
            
            # 加载按钮
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("📂 加载选中知识库", use_container_width=True):
                    if selected_displays:
                        # 加载第一个选中的知识库
                        idx = display_names.index(selected_displays[0])
                        selected_dirname = dir_names[idx]
                        st.session_state.current_kb_dir = selected_dirname
                        st.session_state.current_kb = selected_displays[0]
                        st.session_state.selected_kbs = selected_displays
                        load_existing_knowledge_base(selected_dirname)
                    else:
                        st.warning("请先选择知识库")
            
            with col2:
                if st.button("🗑️", use_container_width=True, help="删除选中知识库"):
                    if selected_displays:
                        idx = display_names.index(selected_displays[0])
                        selected_dirname = dir_names[idx]
                        if delete_knowledge_base(selected_dirname):
                            if "current_kb_dir" in st.session_state and st.session_state.current_kb_dir == selected_dirname:
                                del st.session_state.current_kb_dir
                                del st.session_state.current_kb
                                st.session_state.rag_system = None
                                st.session_state.knowledge_base_ready = False
                            time.sleep(1)
                            st.rerun()
        
        # 状态显示
        if st.session_state.knowledge_base_ready:
            st.success(f"✅ 当前知识库: {st.session_state.get('current_kb', '未知')}")
        
        # ==================== 文件管理 ====================
        st.markdown("---")
        st.markdown("**📁 文件管理**")
        
        # 重新获取知识库列表（避免作用域问题）
        file_kb_list = get_knowledge_bases()
        if file_kb_list:
            file_display_names = [item[1] for item in file_kb_list]
            file_dir_names = [item[0] for item in file_kb_list]
            
            # 选择要管理的知识库
            file_mgmt_kb = st.selectbox(
                "选择知识库查看文件",
                options=file_display_names,
                index=file_display_names.index(st.session_state.get('current_kb')) if st.session_state.get('current_kb') in file_display_names else 0,
                key="file_mgmt_kb"
            )
            
            if file_mgmt_kb:
                idx = file_display_names.index(file_mgmt_kb)
                dirname = file_dir_names[idx]
                metadata = get_kb_metadata(dirname)
                
                files = metadata.get("files", [])
                chunk_count = metadata.get("chunk_count", 0)
                created_at = metadata.get("created_at", "未知")
                
                # 显示统计信息
                st.caption(f"📊 {len(files)} 个文件 | {chunk_count} 块 | 创建于 {created_at}")
                
                # 文件列表
                if files:
                    with st.expander(f"📄 文件列表", expanded=False):
                        for i, f in enumerate(files):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{i+1}.** {f}")
                else:
                    st.info("该知识库暂无文件信息（可能是旧版创建的）")
        else:
            st.info("暂无知识库，请先新建")
        
        # 对话管理
        st.markdown("---")
        st.markdown("**💬 对话管理**")
        
        from src.chat_db import create_conversation, add_message, get_messages, get_conversations, delete_conversation
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ 新建对话", use_container_width=True):
                # 创建新对话
                kb_name = st.session_state.get('current_kb', '未知')
                new_id = create_conversation(kb_name)
                st.session_state.current_conversation_id = new_id
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_conversation_id = None
                st.rerun()
        
        # 显示历史对话（从数据库加载）
        from src.chat_db import rename_conversation
        
        conversations = get_conversations(limit=10)
        history_count = len(conversations)
        
        with st.expander(f"📜 历史对话 ({history_count})", expanded=True):
            if history_count > 0:
                # 构建选项
                conv_options = {f"{c['created_at'][5:16]} - {c['title'][:20]}": c['id'] for c in conversations}
                conv_labels = list(conv_options.keys())
                
                selected_conv = st.selectbox(
                    "选择对话",
                    options=conv_labels,
                    key="history_select",
                    label_visibility="collapsed"
                )
                
                if selected_conv:
                    selected_id = conv_options[selected_conv]
                    
                    # 操作按钮
                    col_load, col_rename, col_del = st.columns([2, 1, 1])
                    
                    with col_load:
                        if st.button("📂 加载", key="load_history", use_container_width=True):
                            st.session_state.messages = get_messages(selected_id)
                            st.session_state.current_conversation_id = selected_id
                            st.rerun()
                    
                    with col_rename:
                        if st.button("✏️", key="rename_btn", help="重命名"):
                            st.session_state.show_rename = selected_id
                    
                    with col_del:
                        if st.button("🗑️", key="delete_btn", help="删除"):
                            delete_conversation(selected_id)
                            st.rerun()
                    
                    # 重命名输入框
                    if st.session_state.get("show_rename") == selected_id:
                        new_name = st.text_input("新名称", key="rename_input")
                        if st.button("确认重命名", key="confirm_rename"):
                            if new_name:
                                rename_conversation(selected_id, new_name)
                                st.session_state.show_rename = None
                                st.rerun()
            else:
                st.caption("暂无历史对话")
        
        # 关于信息
        st.markdown("---")
        st.markdown("""
        <div style="color: #9CA3AF; font-size: 0.8rem;">
        <strong>ScholarRAG</strong> v1.1<br>
        科研论文智能问答助手<br><br>
        技术栈：LangChain + FAISS + Qwen
        </div>
        """, unsafe_allow_html=True)


def build_knowledge_base(uploaded_files, kb_name):
    """建立知识库"""
    from src.ingest import preprocess_documents, split_documents, create_vectorstore
    from src.document_loader import load_documents_from_directory
    
    # 将中文名转换为安全目录名
    safe_dirname = safe_kb_dirname(kb_name)
    
    # 使用安全目录名存储文档
    docs_dir = Path(f"./data/docs/{safe_dirname}")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存名称映射（用于显示）
    save_kb_name_mapping(safe_dirname, kb_name)
    
    # 保存上传的文件
    progress_bar = st.progress(0, text="正在保存上传的文件...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_path = docs_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        progress_bar.progress((i + 1) / len(uploaded_files) * 0.2, 
                              text=f"保存文件中... ({i + 1}/{len(uploaded_files)})")
    
    # 加载所有格式的文档
    progress_bar.progress(0.25, text="正在加载文档...")
    documents = load_documents_from_directory(str(docs_dir))
    
    if not documents:
        st.error("❌ 未能加载任何文档")
        return
    
    # 清洗文档
    progress_bar.progress(0.4, text="正在清洗文档...")
    cleaned_docs = preprocess_documents(documents)
    
    # 切分文档
    progress_bar.progress(0.5, text="正在切分文档...")
    chunks = split_documents(cleaned_docs)
    
    if not chunks:
        st.error("❌ 文档切分失败")
        return
    
    # 创建向量数据库（分批处理）
    progress_bar.progress(0.6, text="正在创建向量数据库...")
    status_text = st.empty()
    total_chunks = len(chunks)
    status_text.text(f"📊 细节：0/{total_chunks} 块 (0%)")
    
    def update_progress(current, total, message):
        """进度回调函数"""
        if total > 0:
            # 进度从 60% 到 90%
            progress = 0.6 + (current / total) * 0.3
            progress_bar.progress(progress, text=message)
            status_text.text(f"📊 细节：{current}/{total} 块 ({int(current/total*100)}%)")
    
    try:
        vectorstore = create_vectorstore(
            chunks, 
            persist_directory=f"./data/{safe_dirname}_db",
            batch_size=20,
            progress_callback=update_progress
        )
        status_text.empty()
        progress_bar.progress(0.95, text="正在初始化问答系统...")
        
        # 保存知识库元数据
        file_names = [f.name for f in uploaded_files]
        save_kb_metadata(safe_dirname, file_names, len(chunks))
        
        # 初始化 RAG 系统
        from src.rag_chain import ScholarRAG
        st.session_state.rag_system = ScholarRAG(persist_directory=f"./data/{safe_dirname}_db")
        st.session_state.knowledge_base_ready = True
        st.session_state.current_kb = kb_name
        st.session_state.current_kb_dir = safe_dirname
        
        progress_bar.progress(1.0, text="完成！")
        st.success(f"✅ 知识库 '{kb_name}' 建立成功！处理了 {len(chunks)} 个文档片段")
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ 建立知识库失败: {str(e)}")
        progress_bar.empty()


def load_existing_knowledge_base(kb_name):
    """加载现有知识库"""
    with st.spinner(f"正在加载知识库 {kb_name}..."):
        try:
            from src.rag_chain import ScholarRAG
            st.session_state.rag_system = ScholarRAG(persist_directory=f"./data/{kb_name}_db")
            st.session_state.knowledge_base_ready = True
            st.session_state.current_kb = kb_name
            st.success("✅ 知识库加载成功！")
            st.rerun()
        except Exception as e:
            st.error(f"❌ 加载失败: {str(e)}")


# ==================== 主界面 ====================

def render_main():
    """渲染主界面"""
    # 标题
    st.markdown('<h1 class="main-title">📚 ScholarRAG</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #6B7280; margin-bottom: 2rem;">'
        '基于 RAG 技术的科研论文智能问答助手</p>',
        unsafe_allow_html=True
    )
    
    # 状态检查
    if not st.session_state.api_key:
        st.info("👈 请先在侧边栏设置千问 API Key")
        return
    
    if not st.session_state.knowledge_base_ready:
        st.info("👈 请先上传 PDF 文档并建立知识库，或加载现有知识库")
        return
    
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 如果有来源信息，显示折叠栏
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])
    
    # 聊天输入
    if prompt := st.chat_input("输入你的问题，例如：这篇论文的主要贡献是什么？"):
        from src.chat_db import add_message, create_conversation
        
        # 如果没有当前对话，创建一个
        if st.session_state.current_conversation_id is None:
            kb_name = st.session_state.get('current_kb', '未知')
            st.session_state.current_conversation_id = create_conversation(kb_name)
        
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        add_message(st.session_state.current_conversation_id, "user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    answer, sources = get_rag_answer(prompt)
                    st.markdown(answer)
                    render_sources(sources)
                    
                    # 保存到历史记录和数据库
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    add_message(st.session_state.current_conversation_id, "assistant", answer)
                    
                except Exception as e:
                    error_msg = f"❌ 生成回答时出错: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def get_rag_answer(question: str):
    """调用 RAG 系统获取答案"""
    if st.session_state.rag_system is None:
        # 尝试使用当前选择的知识库
        if "current_kb_dir" in st.session_state:
            from src.rag_chain import ScholarRAG
            kb_dir = st.session_state.current_kb_dir
            st.session_state.rag_system = ScholarRAG(persist_directory=f"./data/{kb_dir}_db")
        else:
            raise ValueError("请先选择并加载一个知识库")
    
    # 获取对话历史（排除最后一条用户消息）
    chat_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
    
    # 调用 RAG 系统
    answer, source_docs = st.session_state.rag_system.get_answer(question, chat_history)
    
    # 转换源文档为可序列化格式
    sources = []
    for doc in source_docs:
        sources.append({
            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
            "source": doc.metadata.get("source", "未知来源"),
            "page": doc.metadata.get("page", "?")
        })
    
    return answer, sources


def render_sources(sources: List[dict]):
    """渲染参考文档来源"""
    if not sources:
        return
    
    with st.expander(f"📚 参考文档来源 ({len(sources)} 条)"):
        for i, source in enumerate(sources, 1):
            # 提取文件名
            source_path = source.get("source", "未知来源")
            if "/" in source_path or "\\" in source_path:
                filename = source_path.replace("\\", "/").split("/")[-1]
            else:
                filename = source_path
            
            page = source.get("page", "?")
            content = source.get("content", "")
            
            st.markdown(f"""
            <div class="source-card">
                <strong>📄 {filename}</strong> · Page {page}<br>
                <div style="color: #4B5563; margin-top: 8px; font-size: 0.9rem;">
                    {content}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==================== 主程序入口 ====================

def main():
    """主函数"""
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()

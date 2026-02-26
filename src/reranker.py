"""
Reranker 模块（Ollama 版本）
使用 Ollama 部署的 bge-reranker 模型进行重排序
"""

import httpx
from typing import List, Tuple
from langchain_core.documents import Document


class OllamaReranker:
    """
    使用 Ollama 部署的 Reranker 模型
    通过 prompt 方式实现文档重排序
    """
    
    def __init__(
        self, 
        model_name: str = "dengcao/bge-reranker-v2-m3",
        base_url: str = "http://localhost:11434"
    ):
        """
        初始化 Ollama Reranker
        
        Args:
            model_name: Ollama 中的 reranker 模型名称
            base_url: Ollama 服务地址
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = 60.0
        print(f"🔄 初始化 Ollama Reranker ({model_name})...")
    
    def _get_relevance_score(self, query: str, document: str) -> float:
        """
        获取单个文档的相关性分数
        
        Reranker 模型通过 generate 接口返回相关性分数
        """
        try:
            # BGE reranker 使用特定的 prompt 格式
            prompt = f"Query: {query}\nDocument: {document[:1000]}\nRelevance:"
            
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 10,
                        "temperature": 0
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # 尝试从响应中解析分数
                response_text = result.get("response", "0").strip()
                try:
                    # 尝试提取数字
                    import re
                    numbers = re.findall(r"[-+]?\d*\.?\d+", response_text)
                    if numbers:
                        return float(numbers[0])
                except:
                    pass
            return 0.0
        except Exception as e:
            print(f"    ⚠️ 评分失败: {e}")
            return 0.0
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 10
    ) -> List[Document]:
        """
        对文档列表进行重排序
        
        Args:
            query: 用户查询
            documents: 待排序的文档列表
            top_k: 返回前 K 个最相关的文档
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            top_k = len(documents)
        
        print(f"  🔄 Reranker 开始处理 {len(documents)} 个文档...")
        
        # 计算每个文档的相关性分数
        doc_scores: List[Tuple[Document, float]] = []
        for i, doc in enumerate(documents):
            score = self._get_relevance_score(query, doc.page_content)
            doc_scores.append((doc, score))
            if (i + 1) % 10 == 0:
                print(f"    已处理 {i + 1}/{len(documents)} 个文档")
        
        # 按分数排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 top_k 个文档
        reranked_docs = [doc for doc, _ in doc_scores[:top_k]]
        
        print(f"  ✅ Reranker: {len(documents)} → {len(reranked_docs)} 个文档")
        
        return reranked_docs


# 使用 LLM 作为 Reranker（直接 API 调用，避免 PyTorch 依赖）
class LLMReranker:
    """
    使用 Qwen LLM 进行文档重排序
    通过直接 HTTP API 调用实现，避免 PyTorch 依赖问题
    """
    
    def __init__(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = os.getenv("QWEN_API_KEY")
        self.api_base = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = os.getenv("MODEL_NAME", "qwen-plus")
        print("🔄 初始化 LLM Reranker (使用 Qwen API)...")
    
    def _call_qwen(self, prompt: str) -> str:
        """直接调用 Qwen API"""
        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=60.0
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"    ⚠️ Qwen API 错误: {response.status_code}")
                return ""
        except Exception as e:
            print(f"    ⚠️ Qwen API 调用失败: {e}")
            return ""
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 10
    ) -> List[Document]:
        """使用 LLM 对文档进行相关性评分和排序"""
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        print(f"  🔄 LLM Reranker 处理 {len(documents)} 个文档...")
        
        # 文件类型权重 - 学术论文 PDF 优先
        FILE_TYPE_WEIGHTS = {
            "pdf": 1.0,     # PDF 论文优先级最高
            "docx": 0.9,    # Word 文档
            "doc": 0.9,
            "txt": 0.85,    # 纯文本
            "md": 0.85,     # Markdown
            "pptx": 0.75,   # PPT 优先级较低（内容碎片化）
            "ppt": 0.75,
        }
        
        # 批量评分提示
        prompt = f"""请根据问题与文档的相关性进行评分（0-10分）。

问题：{query}

请对以下文档评分，只输出数字，每个文档一行：
"""
        for i, doc in enumerate(documents[:20]):  # 最多处理20个
            prompt += f"\n文档{i+1}：{doc.page_content[:200]}..."
        
        prompt += "\n\n请按顺序输出每个文档的分数（每行一个数字，只输出数字）："
        
        try:
            response_text = self._call_qwen(prompt)
            
            # 解析分数
            import re
            scores = re.findall(r"\d+\.?\d*", response_text)
            scores = [float(s) for s in scores[:len(documents)]]
            
            # 补齐分数
            while len(scores) < len(documents):
                scores.append(0)
            
            # 应用文件类型权重
            weighted_scores = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                file_type = doc.metadata.get("file_type", "pdf").lower()
                weight = FILE_TYPE_WEIGHTS.get(file_type, 0.8)
                weighted_score = score * weight
                weighted_scores.append(weighted_score)
                if i < 5:  # 只打印前5个的权重调整
                    print(f"    [{i+1}] {file_type}: {score:.1f} × {weight} = {weighted_score:.1f}")
            
            # 排序
            doc_scores = list(zip(documents, weighted_scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            reranked = [doc for doc, _ in doc_scores[:top_k]]
            print(f"  ✅ LLM Reranker: {len(documents)} → {len(reranked)} 个文档 (含文件类型权重)")
            return reranked
            
        except Exception as e:
            print(f"  ⚠️ LLM Reranker 失败: {e}，返回原始文档")
            return documents[:top_k]


# 全局 Reranker 实例（延迟加载）
_reranker_instance = None

def get_reranker():
    """获取全局 Reranker 实例"""
    global _reranker_instance
    if _reranker_instance is None:
        # 使用 LLM Reranker（Qwen）- 更可靠
        print("ℹ️ 使用 LLM Reranker (Qwen) 进行文档重排序")
        _reranker_instance = LLMReranker()
    return _reranker_instance


def rerank_documents(
    query: str, 
    documents: List[Document], 
    top_k: int = 10
) -> List[Document]:
    """
    便捷函数：对文档进行重排序
    """
    reranker = get_reranker()
    return reranker.rerank(query, documents, top_k)

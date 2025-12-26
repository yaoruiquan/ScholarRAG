"""
Query Expansion 模块
使用 LLM 将用户问题扩展为多个语义相似的查询变体，提高检索召回率
"""

import httpx
import os
import re
from typing import List
from dotenv import load_dotenv

load_dotenv()


class QueryExpander:
    """
    使用 LLM 进行查询扩展
    """
    
    def __init__(self):
        self.api_key = os.getenv("QWEN_API_KEY")
        self.api_base = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = os.getenv("MODEL_NAME", "qwen-plus")
    
    def _call_llm(self, prompt: str) -> str:
        """调用 Qwen API"""
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
                    "temperature": 0.3
                },
                timeout=30.0
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"  ⚠️ Query Expansion LLM 调用失败: {e}")
            return ""
    
    def expand(self, query: str, num_variants: int = 3) -> List[str]:
        """
        将用户问题扩展为多个语义相似的查询变体
        
        Args:
            query: 原始用户问题
            num_variants: 生成变体数量
            
        Returns:
            包含原始查询和扩展查询的列表
        """
        prompt = f"""请将以下问题扩展为{num_variants}个语义相似但表述不同的查询，用于学术论文检索。
这些查询应该覆盖同一问题的不同表达方式和同义词。

原问题：{query}

请直接输出扩展后的查询，每行一个（不要编号，不要解释）："""
        
        response = self._call_llm(prompt)
        
        if not response:
            return [query]
        
        # 解析扩展查询
        expanded = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # 去除可能的编号前缀
            line = re.sub(r"^[\d]+[\.、\)\]]\s*", "", line)
            if line and len(line) > 2:
                expanded.append(line)
        
        # 确保原始查询在列表中
        if query not in expanded:
            expanded.insert(0, query)
        
        print(f"  🔄 Query Expansion: 1 → {len(expanded)} 个查询")
        for i, q in enumerate(expanded):
            print(f"    [{i+1}] {q}")
        
        return expanded[:num_variants + 1]  # 原始 + N 个变体


# 全局实例
_expander = None

def get_query_expander() -> QueryExpander:
    """获取全局 QueryExpander 实例"""
    global _expander
    if _expander is None:
        _expander = QueryExpander()
    return _expander


def expand_query(query: str, num_variants: int = 3) -> List[str]:
    """便捷函数：扩展查询"""
    expander = get_query_expander()
    return expander.expand(query, num_variants)

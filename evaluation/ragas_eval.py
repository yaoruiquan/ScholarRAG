"""
RAGAS 评估模块
使用 RAGAS 库评估 ScholarRAG 系统的检索和生成质量
配置为使用千问 (Qwen) 作为评估 LLM
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 配置 Qwen LLM 用于 RAGAS 评估
def get_eval_llm():
    """获取用于 RAGAS 评估的 LLM"""
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        temperature=0
    )

def get_eval_embeddings():
    """获取用于 RAGAS 评估的 Embeddings"""
    # 使用 Ollama embeddings
    from src.ingest import CustomOllamaEmbeddings
    return CustomOllamaEmbeddings()


class SimpleEvaluator:
    """
    简化版 RAG 评估器
    不依赖 RAGAS 的自动评估，使用 LLM-as-Judge 方式
    """
    
    EVAL_PROMPT = """请评估以下 RAG 系统的回答质量。

【用户问题】
{question}

【检索到的上下文】
{context}

【系统回答】
{answer}

请从以下维度评分（1-5分）：
1. 忠实度 (Faithfulness): 回答是否完全基于上下文，没有编造信息
2. 相关性 (Relevance): 回答是否切题，直接回答了用户问题
3. 完整性 (Completeness): 回答是否涵盖了上下文中的关键信息
4. 连贯性 (Coherence): 回答是否表达清晰、逻辑通顺

请以 JSON 格式输出：
{{"faithfulness": X, "relevance": X, "completeness": X, "coherence": X, "overall": X, "comment": "简短评价"}}
"""
    
    def __init__(self, rag_system=None):
        self.rag_system = rag_system
        self.llm = get_eval_llm()
    
    def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> Dict:
        """评估单个问答对"""
        context_text = "\n\n---\n\n".join(contexts[:3])  # 只取前3个上下文
        
        prompt = self.EVAL_PROMPT.format(
            question=question,
            context=context_text[:2000],  # 限制长度
            answer=answer
        )
        
        try:
            response = self.llm.invoke(prompt)
            # 尝试解析 JSON
            import re
            json_match = re.search(r'\{[^}]+\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"  评估失败: {e}")
        
        return {"error": "评估失败"}
    
    def run_rag_on_questions(self, questions: List[str]) -> List[Dict]:
        """对问题列表运行 RAG 系统"""
        if self.rag_system is None:
            raise ValueError("RAG system not initialized")
        
        results = []
        for i, q in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {q[:30]}...")
            answer, docs = self.rag_system.get_answer(q)
            contexts = [doc.page_content for doc in docs]
            results.append({
                "question": q,
                "answer": answer,
                "contexts": contexts
            })
        return results
    
    def evaluate_batch(self, questions: List[str]) -> Dict:
        """批量评估"""
        print(f"📊 开始评估 {len(questions)} 个问题...")
        
        # 运行 RAG
        rag_results = self.run_rag_on_questions(questions)
        
        # LLM 评估
        print("🔍 运行 LLM 评估...")
        scores = []
        for r in rag_results:
            score = self.evaluate_single(r["question"], r["answer"], r["contexts"])
            scores.append(score)
            r["scores"] = score
        
        # 计算平均分
        avg_scores = {}
        for key in ["faithfulness", "relevance", "completeness", "coherence", "overall"]:
            valid_scores = [s.get(key, 0) for s in scores if isinstance(s.get(key), (int, float))]
            if valid_scores:
                avg_scores[key] = sum(valid_scores) / len(valid_scores)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(questions),
            "avg_scores": avg_scores,
            "details": rag_results
        }
    
    def generate_report(self, eval_result: Dict, output_path: str = None) -> str:
        """生成评估报告"""
        avg = eval_result.get("avg_scores", {})
        
        report = f"""
==================== RAG 评估报告 ====================
评测时间: {eval_result['timestamp']}
测试样本数: {eval_result['num_samples']}

【平均得分】(满分 5 分)
  忠实度 (Faithfulness):  {avg.get('faithfulness', 'N/A'):.2f}
  相关性 (Relevance):     {avg.get('relevance', 'N/A'):.2f}
  完整性 (Completeness):  {avg.get('completeness', 'N/A'):.2f}
  连贯性 (Coherence):     {avg.get('coherence', 'N/A'):.2f}
  综合得分 (Overall):     {avg.get('overall', 'N/A'):.2f}

========================================================
"""
        print(report)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
                f.write("\n\n【详细结果】\n")
                json.dump(eval_result["details"], f, ensure_ascii=False, indent=2)
            print(f"📄 报告已保存到: {output_path}")
        
        return report


def quick_evaluate(rag_system, questions: List[str]) -> Dict:
    """
    快速评估函数
    
    Args:
        rag_system: ScholarRAG 实例
        questions: 测试问题列表
    
    Returns:
        评估结果字典
    """
    evaluator = SimpleEvaluator(rag_system)
    result = evaluator.evaluate_batch(questions)
    evaluator.generate_report(result)
    return result


def run_full_evaluation(kb_path: str = None, output_path: str = "./evaluation/reports/latest.txt"):
    """
    运行完整评估
    
    Args:
        kb_path: 知识库路径（可选）
        output_path: 报告输出路径
    """
    from src.rag_chain import ScholarRAG
    
    # 加载测试集
    test_set_path = Path(__file__).parent / "data" / "test_set.json"
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    questions = [item["question"] for item in test_set]
    
    # 初始化 RAG
    if kb_path:
        rag = ScholarRAG(persist_directory=kb_path)
    else:
        rag = ScholarRAG()
    
    # 评估
    evaluator = SimpleEvaluator(rag)
    result = evaluator.evaluate_batch(questions)
    evaluator.generate_report(result, output_path)
    
    return result


if __name__ == "__main__":
    # 测试用例
    from src.rag_chain import ScholarRAG
    
    # 初始化 RAG
    rag = ScholarRAG(persist_directory="./data/kb_ee2fb36de231_db")
    
    # 测试问题
    test_questions = [
        "毫米波雷达的工作原理是什么？",
        "无人机目标检测有哪些主要方法？",
    ]
    
    # 快速评估
    quick_evaluate(rag, test_questions)

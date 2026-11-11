
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from settings import rag_summarize_prompt_path, product_expert_prompt



#产品经理提示词
def load_product_expert_prompt():
    try:
        product_expert = get_abs_path(product_expert_prompt)
    except KeyError as e:
        logger.error(f"[load_product_expert_prompt]在配置项中没有product_expert_prompt配置项")
        raise e

    try:
        return open(product_expert, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_product_expert_prompt]解析报告生成提示词出错，{str(e)}")
        raise e

# RAG提示词
def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(rag_summarize_prompt_path)
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在settings配置项中没有 rag_summarize_prompt_path 配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG提示词出错，{str(e)}")
        raise e




if __name__=='__main__':
    print(load_product_expert_prompt())

    
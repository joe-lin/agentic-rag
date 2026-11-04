
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from settings import main_prompt_path, rag_summarize_prompt_path, report_prompt_path, product_expert_prompt

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(main_prompt_path)
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(rag_summarize_prompt_path)
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt = get_abs_path(report_prompt_path)
        # print(report_prompt)
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


def load_product_expert_prompts():
    try:
        product_expert = get_abs_path(product_expert_prompt)
    except KeyError as e:
        logger.error(f"[load_product_expert_prompts]在配置项中没有product_expert_prompt配置项")
        raise e

    try:
        return open(product_expert, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_product_expert_prompts]解析报告生成提示词出错，{str(e)}")
        raise e

if __name__=='__main__':
    print(load_product_expert_prompts())

    
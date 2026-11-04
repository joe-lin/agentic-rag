# import litellm
from litellm import embedding
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from settings import  embedding_model, embedding_api_key, embedding_base_url


# openai/ 前缀 = 走 OpenAI 兼容协议；EMBEDDING_BASE_URL 必须指向真正托管 bge-m3 的服务
response = embedding(
    model=f"openai/{embedding_model}",
    input=["认识oceanbase"],
    api_base=embedding_base_url,
    api_key=embedding_api_key
)
# litellm 返回的 data 是列表，每个元素含 embedding 字段
emb = response["data"][0]["embedding"]
print("向量维度:", len(emb))
# print("前 10 维:", emb[:10])          # 只看头部，避免刷屏
print("完整向量:", emb)               # 1024 维，需要全量就打印这个

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from praisonaiagents.llm.llm import LLM

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from settings import praison_chat_model, praison_api_key, praison_base_url, embedding_model, embedding_api_key, embedding_base_url, chat_model, chat_api_key, chat_base_url
# from langchain_core.embeddings import Embeddings
# import litellm


# class LiteLLMEmbeddings(Embeddings):
#     """用 litellm.embedding 作底层的 LangChain Embeddings 适配器，
#     可直接作为 Chroma 的 embedding_function（需提供 embed_documents / embed_query）。"""
#     def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
#         # litellm 需要 provider 前缀；OpenAI 兼容服务统一加 openai/ 前缀
#         self.model = model if "/" in model else f"openai/{model}"
#         self.api_key = api_key
#         self.base_url = base_url

#     def _embed(self, texts):
#         resp = litellm.embedding(
#             model=self.model,
#             input=texts,
#             api_base=self.base_url,
#             api_key=self.api_key,
#         )
#         # litellm 返回的 data 是列表，每项为 {embedding: [...]}（也支持属性访问）
#         return [item["embedding"] for item in resp["data"]]

#     def embed_documents(self, texts):
#         return self._embed(texts)

#     def embed_query(self, text):
#         return self._embed([text])[0]

embed_model = OpenAIEmbeddings(
    model=embedding_model,
    api_key=embedding_api_key,
    base_url=embedding_base_url
)

chat_model = ChatOpenAI(
    model=chat_model,
    api_key=chat_api_key,
    base_url=chat_base_url
)

praison_llm = LLM(
     model=praison_chat_model
    ,api_key=praison_api_key
    ,base_url=praison_base_url
    )

if __name__ == '__main__':
    # praison_llm.get_response("认识oceanbase")

    print(len(embed_model.embed_query("认识oceanbase")))
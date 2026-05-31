from typing import List
import os
import requests
import numpy as np
import json
from tqdm import tqdm


class Embedder():
    def __init__(self):
        self.YC_API_KEY = os.environ.get("YC_API_KEY")
        self.FOLDER_ID = os.environ.get("YC_FOLDER_ID")

        if not self.YC_API_KEY or not self.FOLDER_ID:
            raise ValueError("Не найдены переменные окружения YC_API_KEY или YC_FOLDER_ID")

        self.doc_uri = f"emb://{self.FOLDER_ID}/text-search-doc/latest"
        self.query_uri = f"emb://{self.FOLDER_ID}/text-search-query/latest"
        self.embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.YC_API_KEY}",
            "x-folder-id": f"{self.FOLDER_ID}"
        }

    def get_embedding(self, text: str, text_type: str):
        data = {
            "modelUri": self.doc_uri if text_type == "doc" else self.query_uri,
            "text": text,
        }
        return np.array(requests.post(self.embed_url, json=data, headers=self.headers).json()["embedding"])

    def embed_documents(self, doc_texts: List[str]):
        text_type = "doc"
        return np.array([self.get_embedding(doc_text, text_type) for doc_text in tqdm(doc_texts, "Embedding documents")])

    def save(self, path, data):
        # Сохраняем np эмбеддинги локально
        #self.store.save(path, data)
        np.save(path, data)

#
# with open("../../data/chunks.json", 'r', encoding="utf-8") as file:
#     data = json.load(file)
#
#
# if __name__ == "__main__":
#     embedder = Embedder()
#     #doc_embeddings = embedder.get_embedding(docs, text_type)
#     docs = [chunk['content'] for chunk in data]
#     docs_embeddings = embedder.embed_documents(docs)
#     embedder.save("docs_embeddings.npy",docs_embeddings)

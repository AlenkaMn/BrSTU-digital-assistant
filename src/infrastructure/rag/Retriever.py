import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cdist

from src.infrastructure.rag.Embedder import Embedder


class Retriever:
    """
    Гибридный Retriever для RAG-системы.

    Использует:
    1. Семантический поиск по эмбеддингам.
    2. Лексический поиск BM25.
    3. Объединение результатов методом Борда.
    """

    def __init__(self, embedded_kb, chunks):
        self.embedder = Embedder()

        self.embedded_kb = np.asarray(embedded_kb)
        self.chunks = np.array(chunks, dtype=object)

        self.docs = [
            self._get_chunk_text(chunk)
            for chunk in self.chunks
        ]

        tokenized_docs = [
            self._tokenize(doc)
            for doc in self.docs
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, k_chunks: int = 5):
        """
        Выполняет гибридный поиск по запросу пользователя.

        :param query: запрос пользователя
        :param k_chunks: количество возвращаемых чанков
        :return: массив наиболее релевантных чанков
        """

        if not query or not query.strip():
            return self.chunks[:0]

        query = query.strip().lower()

        semantic_rank = self._semantic_search(query)
        bm25_rank = self._bm25_search(query)

        final_indices = self._borda_fusion(
            semantic_rank=semantic_rank,
            bm25_rank=bm25_rank,
            k_chunks=k_chunks
        )

        return self.chunks[final_indices]

    def _semantic_search(self, query: str):
        """
        Семантический поиск по эмбеддингам.
        Возвращает индексы чанков, отсортированные от наиболее релевантного.
        """

        emb_query = self.embedder.get_embedding(query, text_type="query")

        dist = cdist(
            emb_query[None, :],
            self.embedded_kb,
            metric="cosine"
        )

        sims = 1 - dist[0]

        semantic_rank = np.argsort(sims)[::-1]

        return semantic_rank

    def _bm25_search(self, query: str):
        """
        Лексический поиск BM25.
        Учитывает точные совпадения слов, названий документов,
        аудиторий, подразделений и других терминов.
        """

        tokenized_query = self._tokenize(query)

        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_rank = np.argsort(bm25_scores)[::-1]

        return bm25_rank

    def _borda_fusion(self, semantic_rank, bm25_rank, k_chunks: int):
        """
        Объединяет результаты семантического поиска и BM25 методом Борда.

        Идея:
        - чем выше документ в отдельном ранжировании, тем больше баллов он получает;
        - итоговый рейтинг строится по сумме баллов из двух поисков.
        """

        scores = defaultdict(float)

        n_docs = len(self.chunks)

        for position, doc_idx in enumerate(semantic_rank):
            scores[int(doc_idx)] += n_docs - position

        for position, doc_idx in enumerate(bm25_rank):
            scores[int(doc_idx)] += n_docs - position

        final_rank = sorted(
            scores.keys(),
            key=lambda doc_idx: scores[doc_idx],
            reverse=True
        )

        return final_rank[:k_chunks]

    def _get_chunk_text(self, chunk) -> str:
        """
        Достаёт текст чанка для BM25.
        """

        if isinstance(chunk, dict):
            return str(chunk.get("content", "")).lower()

        return str(chunk).lower()

    def _tokenize(self, text: str) -> list[str]:
        """
        Токенизация для BM25.

        Стоп-слова НЕ удаляются, чтобы сохранить семантику запроса.
        """

        text = text.lower()

        # Удаляем последовательности цифр, как в твоём описании Rewriter
        text = re.sub(r"\d+", " ", text)

        # Оставляем русские/английские слова
        tokens = re.findall(r"[a-zа-яё]+", text)

        return tokens

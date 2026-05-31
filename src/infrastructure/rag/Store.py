import json

from src.infrastructure.rag.Rewriter import Rewriter


class Store:
    def __init__(self):
        self.rewriter = Rewriter()

    def create_docs(self, chunks_path):
        with open(chunks_path, 'r', encoding="utf-8") as file:
            data = json.load(file)
        return data

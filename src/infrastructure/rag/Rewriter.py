import re


class Rewriter:
    """
    Модуль предварительной обработки пользовательского запроса.

    Выполняет мягкую нормализацию текста перед передачей запроса
    в Embedder и Retriever.
    """

    def __init__(self):
        pass

    def rewrite(self, query: str) -> str:
        """
        Нормализует пользовательский запрос.
        """

        if query is None:
            return ""

        query = query.strip()

        if not query:
            return ""

        # Приведение к нижнему регистру
        query = query.lower()

        # Удаление последовательностей цифр
        query = re.sub(r"\d+", " ", query)

        # Удаление лишних пробелов
        query = re.sub(r"\s+", " ", query)

        return query.strip()
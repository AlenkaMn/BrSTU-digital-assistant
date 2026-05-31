import json
import logging
import os

import numpy as np

from src.infrastructure.data_base import ChatDB
from src.infrastructure.llm.YandexLLMClient import YandexLLMClient
from src.infrastructure.rag import Embedder, Retriever, Store
from src.infrastructure.rag.Rewriter import Rewriter
from src.resources import MAIN_PROMPT_TEMPLATE


# =========================
# Настройка логирования
# =========================

LOG_DIR = "../logs"
LOG_FILE = "workflow_processor.log"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


logger.info("Текущая рабочая директория: %s", os.getcwd())


# =========================
# Инициализация модулей
# =========================

try:
    logger.info("Инициализация Embedder, Rewriter, Store")

    embedder = Embedder()
    rewriter = Rewriter()
    store = Store()

    logger.info("Загрузка чанков из ../data/chunks.json")
    chunks = store.create_docs("../data/chunks.json")
    docs = [chunk["content"].lower() for chunk in chunks]

    logger.info("Количество загруженных чанков: %s", len(chunks))

    # Если нужно пересчитать эмбеддинги:
    # docs_embeddings = embedder.embed_documents(docs)
    # embedder.save("../data/docs_embedding.npy", docs_embeddings)

    logger.info("Загрузка эмбеддингов документов из ../data/docs_embedding.npy")
    docs_embeddings = np.load("../data/docs_embedding.npy")

    logger.info("Размерность эмбеддингов документов: %s", docs_embeddings.shape)

    retriever = Retriever(docs_embeddings, chunks)
    yandex_llm_client = YandexLLMClient()
    db = ChatDB("sqlite:///chat_history_test.db")

    logger.info("Workflow processor успешно инициализирован")

except Exception as e:
    logger.exception("Ошибка при инициализации workflow_processor: %s", e)
    raise


def format_history(history: list) -> str:
    """
    Формирует текстовую историю диалога из новой таблицы messages.
    """

    logger.info("Формирование истории диалога. Получено сообщений: %s", len(history))

    history_str = ""

    for msg in history[-5:]:
        logger.debug(
            "История | created_at=%s | user_message=%s | system_answer=%s",
            msg.created_at,
            msg.user_message,
            msg.system_answer
        )

        if msg.user_message:
            history_str += f"Пользователь: {msg.user_message}\n"

        if msg.system_answer:
            history_str += f"Ассистент: {msg.system_answer}\n"

    logger.info("История диалога сформирована. Длина: %s символов", len(history_str))

    return history_str.strip()


def parse_llm_json_response(llm_response: str) -> dict:
    """
    Парсит JSON-ответ от LLM.
    """

    logger.info("Парсинг JSON-ответа от LLM")

    cleaned_response = llm_response.replace("`", "").strip()

    if cleaned_response.lower().startswith("json"):
        cleaned_response = cleaned_response[4:].strip()

    try:
        parsed_response = json.loads(cleaned_response)
        logger.info("JSON-ответ от LLM успешно распарсен")
        return parsed_response

    except json.JSONDecodeError as e:
        logger.error("Не удалось распарсить JSON-ответ от LLM")
        logger.error("Ответ LLM: %s", cleaned_response)
        logger.exception("JSONDecodeError: %s", e)
        raise


def collect_images_from_chunks(relevant_chunks: list, chunk_used_ids: list) -> list[str]:
    """
    Собирает изображения из metadata чанков, реально использованных LLM.
    """

    logger.info("Сбор изображений из использованных чанков")
    logger.info("Использованные LLM chunk_id: %s", chunk_used_ids)

    images = []

    chunks_by_id = {
        chunk["metadata"]["id"]: chunk
        for chunk in relevant_chunks
    }

    for chunk_id in chunk_used_ids:
        original_chunk_id = chunk_id

        if chunk_id not in chunks_by_id:
            try:
                chunk_id = int(chunk_id)
            except Exception:
                logger.warning(
                    "chunk_id=%s отсутствует в chunks_by_id и не может быть преобразован к int",
                    original_chunk_id
                )
                continue

        if chunk_id not in chunks_by_id:
            logger.warning(
                "chunk_id=%s не найден среди релевантных чанков",
                chunk_id
            )
            continue

        metadata = chunks_by_id[chunk_id].get("metadata", {})

        if "images" in metadata and metadata["images"] != "None":
            image_path = f'../images/{metadata["images"]}'
            images.append(image_path)
            logger.info("Добавлено изображение: %s", image_path)

    logger.info("Всего найдено изображений: %s", len(images))

    return images


def get_answer(
        llm_client,
        message,
        history: list,
        user_id: int = None,
        is_toxic: bool = False,
        is_social_query: bool = False,
        user_mood: str = None,
        save_to_db: bool = True
):
    """
    Основной workflow обработки пользовательского запроса.

    Порядок работы:
    1. Формирование истории диалога.
    2. Нормализация запроса через Rewriter.
    3. Поиск релевантных чанков через Retriever.
    4. Формирование промпта и обращение к LLM.
    5. Парсинг JSON-ответа.
    6. Сбор изображений из использованных чанков.
    7. Сохранение результата в БД.
    """

    logger.info("Начало обработки пользовательского запроса")

    try:
        user_message = message.text

        logger.info("user_id=%s", user_id)
        logger.info("Исходный запрос пользователя: %s", user_message)
        logger.info(
            "Результаты классификации | is_toxic=%s | is_social_query=%s | user_mood=%s",
            is_toxic,
            is_social_query,
            user_mood
        )

        # 1. Формируем историю для LLM
        history_str = format_history(history)

        # 2. Нормализуем запрос через Rewriter
        query = rewriter.rewrite(user_message)
        logger.info("Нормализованный запрос: %s", query)

        # 3. Ищем релевантные чанки через гибридный Retriever
        logger.info("Запуск Retriever")
        relevant_chunks = retriever.search(query, 5)

        logger.info("Retriever вернул чанков: %s", len(relevant_chunks))

        for chunk in relevant_chunks:
            logger.debug(
                "Релевантный chunk_id=%s | content_preview=%s",
                chunk["metadata"]["id"],
                chunk["content"][:300]
            )

        chunks_info = {
            chunk["metadata"]["id"]: chunk["content"]
            for chunk in relevant_chunks
        }

        logger.info("Сформирован chunks_info для LLM. Количество чанков: %s", len(chunks_info))

        # 4. Получаем ответ от LLM
        logger.info("Отправка запроса в LLM")

        llm_response = llm_client.get_response(
            MAIN_PROMPT_TEMPLATE.format(
                question=query,
                chunks_info=chunks_info,
                history=history_str
            )
        )

        logger.info("Ответ от LLM получен")
        logger.debug("Сырой ответ LLM: %s", llm_response)

        # 5. Парсим JSON-ответ
        parsed_llm_response = parse_llm_json_response(llm_response)

        answer = parsed_llm_response["answer"]
        chunk_used_ids = parsed_llm_response.get("usable_chunk_ids", [])

        logger.info("Сформированный ответ: %s", answer)
        logger.info("LLM использовала chunk_id: %s", chunk_used_ids)

        # 6. Собираем изображения
        images = collect_images_from_chunks(relevant_chunks, chunk_used_ids)

        # 7. Сохраняем диалог в БД
        if save_to_db and user_id is not None:
            logger.info("Сохранение сообщения и ответа в БД")

            db.add_message(
                user_id=user_id,
                user_message=user_message,
                system_answer=answer,
                is_toxic=is_toxic,
                is_social_query=is_social_query,
                user_mood=user_mood
            )

            logger.info("Сообщение успешно сохранено в БД")

        elif save_to_db and user_id is None:
            logger.warning("save_to_db=True, но user_id=None. Сообщение не сохранено в БД")

        logger.info("Обработка пользовательского запроса завершена успешно")

        return answer, images

    except Exception as e:
        logger.exception("Ошибка при обработке пользовательского запроса: %s", e)
        raise
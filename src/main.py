import asyncio
import logging
import os

from adapters.tg.bot import create_bot_dispatcher
from infrastructure.data_base import ChatDB
from infrastructure.llm.YandexLLMClient import YandexLLMClient

from dotenv import load_dotenv


load_dotenv()


LOG_DIR = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "bot.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Запуск Telegram-бота")



async def main():
    try:
        logger.info("Инициализация LLM клиента и базы данных")
        client = YandexLLMClient()
        chat_db = ChatDB()

        logger.info("Создание диспетчера бота")
        dp, bot = create_bot_dispatcher(
            "8444520246:AAExAfhdRKjk4MsuKKu0XucwNK18Wv0I_fM",
            client,
            chat_db
        )

        logger.info("Запуск polling Telegram-бота")
        await dp.start_polling(bot)

    except Exception as e:
        logger.exception("Ошибка при запуске Telegram-бота: %s", e)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
        logger.info("Telegram-бот завершил работу")
    except Exception as e:
        logger.exception("Программа завершилась с ошибкой: %s", e)

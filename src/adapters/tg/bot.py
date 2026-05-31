from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command

from infrastructure.data_base import ChatDB

from services.workflow_processor import get_answer


def create_bot_dispatcher(bot_token: str, llm_client, chat_db: ChatDB) -> tuple[Dispatcher, Bot]:
    bot = Bot(token=bot_token)
    # он перенаправляет запросы на другие функции в от зависимости от введённого действия пользователя
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: types.Message):
        await message.answer("""
        Здравствуйте!

Я — ИИ-помощник студента БрГТУ. Я могу помочь вам с различными задачами, например:
* ответить на вопросы, связанные с учёбой;
* подсказать информацию о факультетах, специальностях и преподавателях;
* помочь разобраться в расписании занятий;
* предложить полезные ресурсы и материалы для учёбы;
* поддержать диалог на разные темы.

Если у вас есть вопросы или нужна помощь, не стесняйтесь обращаться ко мне. Я буду рад помочь!""")

    @dp.message()
    async def handle_request(message: types.Message):
        print(message)
        chat_db.add_message(message.from_user.id, 'user', message.text)
        # Получаем историю
        history = chat_db.get_last_messages(message.from_user.id, limit=5)
        answer, image_paths = get_answer(llm_client, message, history)
        print(image_paths)
        #выводим текст в диалог
        await message.answer(answer)
        chat_db.add_message(message.from_user.id, 'bot', answer, image_paths)
        for image_path in image_paths:
            try:
                with open(image_path, "rb") as f:
                    from aiogram.types import BufferedInputFile
                    await message.answer_photo(
                        photo=BufferedInputFile(f.read(), filename="image.png")
                    )
            except FileNotFoundError:
                await message.answer("Изображение не найдено.")
            except Exception as e:
                await message.answer(f"Ошибка: {e}")


    return dp, bot

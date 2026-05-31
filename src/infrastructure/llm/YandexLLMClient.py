import os
from src.core.LLMClient import LLMClient
from yandex_gpt import YandexGPT, YandexGPTConfigManagerForAPIKey


class YandexLLMClient(LLMClient):
    def __init__(self):
        super().__init__()

        model_type = "yandexgpt"

        api_key = os.getenv("YC_API_KEY")
        catalog_id = os.getenv("YC_FOLDER_ID")

        if not api_key or not catalog_id:
            raise ValueError("Не найдены переменные окружения YC_API_KEY или YC_FOLDER_ID")

        config = YandexGPTConfigManagerForAPIKey(
            model_type=model_type,
            catalog_id=catalog_id,
            api_key=api_key
        )

        self.client = YandexGPT(config_manager=config)
    def get_response(self, query: str):
        messages = [
            {
                "role": "user",
                "text": query
            }
        ]
        response = self.client.get_sync_completion(messages=messages)
        return response
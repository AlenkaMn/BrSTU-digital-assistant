from abc import ABC, abstractmethod


class LLMClient(ABC):
    def __init__(self):
        self.system_prompt = None
        self.client = None


    @abstractmethod
    def get_response(self, message):
        pass

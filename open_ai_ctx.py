from dotenv import dotenv_values
from langchain_openai import ChatOpenAI


class OpenAICtx:
    """
    This class is a context manager for using AzureAI on-demand.
    It loads the configuration from a dotenv file and initializes an AzureChatOpenAI instance.
    """

    def __init__(self, dotenv_file=None):
        """
        Initialize the context manager.
        Load the configuration from the dotenv file.

        :param dotenv_file: The path to the dotenv file. If None, defaults to ".azure.env".
        """
        if dotenv_file is None:
            dotenv_file = ".env"
        self.config = dotenv_values(dotenv_file)

    def __enter__(self):
        """
        Enter the context.
        Initialize the AzureChatOpenAI instance with the loaded configuration.

        :return: The initialized AzureChatOpenAI instance.
        """
        self.llm = ChatOpenAI(openai_api_key=self.config["OPENAI_API_KEY"],
                              temperature=self.config["OPENAI_API_TEMPERATURE"],
                              model_name=self.config["OPENAI_API_MODEL_NAME"])
        return self.llm

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Exit the context.
        If an exception occurred in the with block, print its type and message.

        :param exc_type: The type of the exception.
        :param exc_value: The instance of the exception.
        :param exc_tb: The traceback of the exception.

        :return: True if an exception occurred, None otherwise.
        """
        print("Leaving the context...")
        if isinstance(exc_value, Exception):
            print(f"An exception occurred in your with block: {exc_type}")
            print(f"Exception message: {exc_value}")
            return True

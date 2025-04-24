from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    translation_model: str = "Helsinki-NLP/opus-mt-en-ROMANCE"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
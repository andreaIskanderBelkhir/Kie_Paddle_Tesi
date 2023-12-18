from pydantic import BaseSettings


class CommonSettings(BaseSettings):
    PROJECT_NAME: str = 'myBiros Platform API'
    PROJECT_DESCRIPTION: str = 'myBiros API for training data generation'
    VERSION: str = 0.1
    DEBUG: bool = True

#9aYH8hQaUBNMVhtf
class DatabaseSettings(BaseSettings):
    DATABASE_URL: str = ""
    DATABASE_NAME: str = ""
    DATASET_FEEDBACK_ID: str=""



class AWSSettings(BaseSettings):
    BUCKET_NAME: str = ""
    BUCKET_FEEDBACK: str = ""
    AWS_ACCESS_KEY_ID: str=""
    AWS_SECRET_ACCESS_KEY: str=""


class EndpointsSettings(BaseSettings):
    TRAINING_API_ENDPOINT: str = "http://127.0.0.1:8080/train"


class Settings(
            CommonSettings,
            DatabaseSettings,
            AWSSettings,
            EndpointsSettings
):
    pass


cfg = Settings()

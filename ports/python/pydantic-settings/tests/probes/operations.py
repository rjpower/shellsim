"""Read a mounted dotenv file with typed settings and source precedence."""

import os

from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Database(BaseModel):
    host: str
    port: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="/work/settings.env", env_prefix="APP_", env_nested_delimiter="__")
    name: str
    workers: int
    enabled: bool
    database: Database


settings = Settings()
assert settings.model_dump() == {
    "name": "mounted guest",
    "workers": 3,
    "enabled": True,
    "database": {"host": "db.guest", "port": 5432},
}
os.environ["APP_WORKERS"] = "5"
os.environ["APP_DATABASE__PORT"] = "6543"
assert Settings().workers == 5 and Settings().database.port == 6543
assert Settings(workers=7).workers == 7
os.environ["APP_WORKERS"] = "invalid"
try:
    Settings()
except ValidationError as error:
    assert error.errors()[0]["type"] == "int_parsing"
else:
    raise AssertionError("invalid typed setting was accepted")

print("pydantic-settings: mounted settings passed")

"""Configuración global cargada desde variables de entorno / `.env`."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inegi_api_token: str = Field(default="", alias="INEGI_API_TOKEN")
    inegi_base_url: str = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
    inegi_geo: str = "0700"  # Nacional
    cache_dir: Path = PROJECT_ROOT / ".cache" / "inegi"
    data_dir: Path = PROJECT_ROOT / "data"

    def require_token(self) -> str:
        if not self.inegi_api_token:
            raise RuntimeError(
                "INEGI_API_TOKEN no está configurado. Crea un .env con tu token "
                "(ver .env.example) o exporta la variable de entorno."
            )
        return self.inegi_api_token


settings = Settings()

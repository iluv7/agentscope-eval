"""Server-side judge and concurrency configuration."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read EVAL_* variables and an optional local .env file."""

    model_config = SettingsConfigDict(
        env_prefix="EVAL_", env_file=".env", extra="ignore"
    )
    judge_model: str = ""
    judge_api_key: SecretStr = SecretStr("")
    judge_base_url: str = ""
    max_concurrent: int = Field(default=4, ge=1, le=32)
    metric_timeout_seconds: float = Field(default=120, gt=0, le=600)

    @property
    def judge_configured(self) -> bool:
        """Whether the model name and API key have both been supplied."""
        return bool(
            self.judge_model.strip()
            and self.judge_api_key.get_secret_value().strip()
        )

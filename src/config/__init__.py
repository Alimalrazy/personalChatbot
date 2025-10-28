try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path

class Config:
    def __init__(self, secret_file: str = "secret.toml"):
        self.secret_file = Path(secret_file)
        self._secrets = None

    @property
    def secrets(self):
        if self._secrets is None:
            with open(self.secret_file, "rb") as f:
                self._secrets = tomllib.load(f)
        return self._secrets

    @property
    def api_key(self):
        return self.secrets["secrets"]["api_key"]

config = Config()

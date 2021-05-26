from strictyaml import load
from pathlib import Path

def load_config_data(path: str) -> dict:
	yaml_string = Path(path).read_text()
	cfg = load(yaml_string, schema=None)
	cfg: dict = cfg.data
	return cfg
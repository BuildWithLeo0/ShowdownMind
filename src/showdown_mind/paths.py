from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
SHOWDOWN_DIR = RUNTIME_DIR / "pokemon-showdown"
SHOWDOWN_LOCK_FILE = CONFIG_DIR / "showdown.lock"
SHOWDOWN_LOCAL_CONFIG = CONFIG_DIR / "showdown.local.js"
REPLAY_DIR = RUNTIME_DIR / "replays"

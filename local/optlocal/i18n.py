"""Small, offline translation catalog shared by the GUI and CLI.

Only display text is translated. Configuration keys, metric names, result
fields, and simulator output retain their original representation.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile

LANGUAGES = {"zh": "中文", "ja": "日本語", "en": "English"}
_language = "zh"


def normalize_language(language):
    code = str(language).strip().lower().split(".", 1)[0].replace("_", "-").split("-", 1)[0]
    if code not in LANGUAGES:
        raise ValueError("Unsupported language: %s (zh, ja, en)" % language)
    return code


def settings_path():
    explicit = os.environ.get("VCAL_SETTINGS_FILE")
    if explicit:
        return Path(explicit).expanduser()
    root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "vcal" / "settings.json"


def _read_settings():
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def configure_language(language=None):
    """Explicit argument > VCAL_LANG > saved preference > Chinese."""
    if language is not None:
        return set_language(language)
    language = os.environ.get("VCAL_LANG")
    if not language:
        try:
            language = normalize_language(_read_settings().get("language", "zh"))
        except ValueError:
            language = "zh"
    return set_language(language)


def set_language(language, persist=False):
    global _language
    code = normalize_language(language)
    if persist:
        path = settings_path()
        settings = _read_settings()
        settings["language"] = code
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".vcal-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(settings, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    _language = code
    return code


def get_language():
    return _language


@lru_cache(maxsize=3)
def _catalog(language):
    path = Path(__file__).with_name("locales") / (language + ".json")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def tr(source, language=None):
    code = get_language() if language is None else normalize_language(language)
    return _catalog(code).get(source, source)

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROMPTS_DIR = os.path.join(_PROJECT_ROOT, "prompts")


def load_prompt(name, default=""):
    path = os.path.join(_PROMPTS_DIR, f"{name}.txt")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().rstrip("\n")
        except OSError:
            pass
    return default


def prompts_dir():
    return _PROMPTS_DIR

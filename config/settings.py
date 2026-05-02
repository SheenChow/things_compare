import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3-flash-preview"

if not GEMINI_API_KEY:
    raise ValueError("请设置环境变量 GEMINI_API_KEY")

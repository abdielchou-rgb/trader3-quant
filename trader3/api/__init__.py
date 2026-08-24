"""3号交易员 — FastAPI 服务 (M6)

`app` 可直接导入启动:  uvicorn trader3.api:app --port 8000
"""

from trader3.api.server import app, get_trader3

__all__ = ["app", "get_trader3"]

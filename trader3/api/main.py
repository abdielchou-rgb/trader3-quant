import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from trader3.v2.live import Order, OrderSide, OrderType, PaperBroker, TimeInForce


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()
broker: PaperBroker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global broker
    broker = PaperBroker(initial_cash=1_000_000.0)
    await broker.connect()
    yield
    await broker.disconnect()


app = FastAPI(
    title="Trader3 Dashboard API",
    description="Quantitative Trading System Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


class OrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"


class FactorRequest(BaseModel):
    factor_name: str
    symbols: list[str]
    params: dict[str, Any] = {}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/account")
async def get_account():
    if not broker:
        return {"error": "Broker not initialized"}
    account = await broker.get_account()
    return {
        "account_id": account.account_id,
        "cash": account.cash,
        "equity": account.equity,
        "buying_power": account.buying_power,
        "positions_count": len(account.positions),
        "updated_at": account.updated_at.isoformat(),
    }


@app.get("/api/positions")
async def get_positions():
    if not broker:
        return {"error": "Broker not initialized"}
    positions = await broker.get_positions()
    return {
        sym: {
            "symbol": pos.symbol,
            "quantity": pos.quantity,
            "avg_cost": pos.avg_cost,
            "market_value": pos.market_value,
            "unrealized_pnl": pos.unrealized_pnl,
            "last_price": pos.last_price,
            "updated_at": pos.updated_at.isoformat(),
        }
        for sym, pos in positions.items()
    }


@app.get("/api/orders")
async def get_orders(status: str | None = None):
    if not broker:
        return {"error": "Broker not initialized"}
    from trader3.v2.live.broker_base import OrderStatus
    order_status = OrderStatus(status) if status else None
    orders = await broker.get_orders(order_status)
    return {
        "orders": [
            {
                "client_order_id": o.client_order_id,
                "broker_order_id": o.broker_order_id,
                "symbol": o.symbol,
                "side": o.side.value,
                "quantity": o.quantity,
                "order_type": o.order_type.value,
                "price": o.price,
                "status": o.status.value,
                "filled_qty": o.filled_qty,
                "avg_fill_price": o.avg_fill_price,
                "created_at": o.created_at.isoformat(),
                "updated_at": o.updated_at.isoformat(),
            }
            for o in orders
        ]
    }


@app.post("/api/orders")
async def place_order(request: OrderRequest):
    if not broker:
        return {"error": "Broker not initialized"}

    order = Order(
        symbol=request.symbol,
        side=OrderSide(request.side),
        quantity=request.quantity,
        order_type=OrderType(request.order_type),
        price=request.price,
        stop_price=request.stop_price,
        time_in_force=TimeInForce(request.time_in_force),
    )

    result = await broker.place_order(order)
    await manager.broadcast({
        "type": "order_update",
        "data": {
            "client_order_id": result.client_order_id,
            "status": result.status.value,
        }
    })
    return {
        "client_order_id": result.client_order_id,
        "broker_order_id": result.broker_order_id,
        "status": result.status.value,
        "filled_qty": result.filled_qty,
        "avg_fill_price": result.avg_fill_price,
    }


@app.delete("/api/orders/{client_order_id}")
async def cancel_order(client_order_id: str):
    if not broker:
        return {"error": "Broker not initialized"}
    success = await broker.cancel_order(client_order_id)
    return {"success": success}


@app.get("/api/market-data")
async def get_market_data(symbols: str):
    if not broker:
        return {"error": "Broker not initialized"}
    sym_list = symbols.split(",")
    data = await broker.get_market_data(sym_list)
    return {
        sym: {
            "symbol": md.symbol,
            "price": md.price,
            "bid": md.bid,
            "ask": md.ask,
            "volume": md.volume,
            "timestamp": md.timestamp.isoformat(),
        }
        for sym, md in data.items()
    }


@app.get("/api/factors")
async def list_factors(category: str | None = None):
    from trader3.v2.factors import list_factors
    factors = list_factors(category)
    return {
        "factors": [
            {
                "name": f.name,
                "description": f.description,
                "category": f.category,
                "params": f.params,
                "formula": f.formula,
            }
            for f in factors
        ]
    }


@app.post("/api/factors/compute")
async def compute_factor(request: FactorRequest):
    import numpy as np
    import pandas as pd

    from trader3.v2.factors import compute_factor

    data = pd.DataFrame({
        "close": np.random.randn(100).cumsum() + 100,
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 101,
        "low": np.random.randn(100).cumsum() + 99,
        "volume": np.random.randint(10000, 100000, 100),
    })

    try:
        result = compute_factor(data, request.factor_name, **request.params)
        return {
            "factor": request.factor_name,
            "values": result.dropna().tolist()[-50:],
            "latest": float(result.iloc[-1]) if not result.empty else None,
        }
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"echo": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_market_updates():
    while True:
        await asyncio.sleep(1.0)
        if broker and manager.active_connections:
            try:
                account = await broker.get_account()
                positions = await broker.get_positions()
                await manager.broadcast({
                    "type": "portfolio_update",
                    "data": {
                        "equity": account.equity,
                        "cash": account.cash,
                        "positions": len(positions),
                        "timestamp": datetime.now().isoformat(),
                    }
                })
            except Exception:
                pass


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_market_updates())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

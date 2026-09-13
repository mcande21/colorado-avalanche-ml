from fastapi import FastAPI

from avalanche_ml.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Colorado Avalanche ML", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()

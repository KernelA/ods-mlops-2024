from fastapi import FastAPI

from .routes.sentiment import router

app = FastAPI()
app.include_router(router)


@app.get("/")
async def root():
    return {"status": "running"}

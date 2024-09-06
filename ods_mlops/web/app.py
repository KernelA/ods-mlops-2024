from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .routes.sentiment import router

app = FastAPI()
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
     <html>
        <head>
            <title>Running</title>
        </head>
        <body>
        <h1>Server is running</h1>
        </body>
    </html>
    """

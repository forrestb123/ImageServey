from os import read
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="web")

@app.get("/fimages/{item_id}")
def get_fimage(item_id: int, response: Response):
    response.content_type = "image/png"
    return open(f"fake_images/{item_id}.png", "rb")

@app.get("/rimages/{item_id}")
def get_fimage(item_id: int, response: Response):
    response.content_type = "image/png"
    return open(f"fake_images/{item_id}.png", "rb")
from fastapi import FastAPI
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from giapi.config import CONFIG
from giapi.routes import learning, preprocessing
from gicommon.models.learning import DLModelInDB
from gicommon.models.preprocessing import DatasetInDB

app = FastAPI()

app.include_router(preprocessing.router)
app.include_router(learning.router)


@app.on_event("startup")
async def app_init():
    app.db = AsyncIOMotorClient(CONFIG.mongo_uri).account
    await init_beanie(app.db, document_models=[
        DatasetInDB,
        DLModelInDB,
    ])

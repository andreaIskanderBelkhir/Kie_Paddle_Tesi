import time

from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from motor.motor_asyncio import AsyncIOMotorClient

from app.kie.inferenza_re_api import router as InfernzaRouter
from app.training.train_api import router as TrainingRouter


app=FastAPI()

app.include_router(InfernzaRouter, prefix="/inferenza",  tags=["inferenza"])
app.include_router(TrainingRouter,prefix="/train", tags=["training"])
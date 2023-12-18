from fastapi import APIRouter, File, UploadFile, status, Form, Query
import app.training.train as Trainer
import os



router = APIRouter()
ser_conf="./app/training/ser_conf.yml"
re_conf="./app/training/re_conf.yml"

@router.post("/trainSER", summary="Perform training")
async def trainingSer():
	os.makedirs("./app/training/train_data", exist_ok=True)
	os.makedirs("./app/training/train_data/train", exist_ok=True)
	os.makedirs("./app/training/train_data/val", exist_ok=True)
	trainer = Trainer.Train(ser_conf)
	trainer()

@router.post("/trainRe", summary="Perform training")
async def trainingRe():
	os.makedirs("./app/training/train_data", exist_ok=True)
	os.makedirs("./app/training/train_data/train", exist_ok=True)
	os.makedirs("./app/training/train_data/val", exist_ok=True)
	trainer = Trainer.Train(re_conf)
	trainer()

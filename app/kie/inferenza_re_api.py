from fastapi import APIRouter, File, UploadFile, status, Form, Query
import app.kie.inferenza_re as RePredictor
from app.apps.tools.formattajson import oldjson, dbjson,formatta_annotation
from app.db.database import add_item,retrieve_dataset,update_dataset
import yaml
import numpy as np
import os
import shutil
import json
import requests
import base64
import cv2
import boto3
from app.config import cfg
from ppocr.data import create_operators, transform
from app.apps.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.visual import draw_re_results
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps, print_dict
from app.apps.core.ocr import OCR

router = APIRouter()
s3 = boto3.client(
    's3',
    aws_access_key_id=cfg.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=cfg.AWS_SECRET_ACCESS_KEY
)

@router.post("/", summary="Perform inferenza")
async def inferenza(ocr_proprietario : bool= Query(False,description="con true usera ocr proprietario"),file: UploadFile=File(...)):
    if ocr_proprietario==True:

        ocr_client = OCR('aws')
        contents = await file.read()

        file0 = await file.seek(0) 
        os.makedirs("./app/kie/upload", exist_ok=True)
        dest = os.path.join("./app/kie/upload", file.filename)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer) 
        s3.put_object(Bucket=cfg.BUCKET_NAME, Key="public/"+cfg.BUCKET_FEEDBACK+file.filename, Body=contents)
        response = ocr_client.detect_text(contents)
        os.makedirs("./app/kie/upload/ocr_result", exist_ok=True)


        ocr_res=[]
        for res in response["bounding_box"]:
            ocr_res.append({
                "transcription": res["text"],
                "points":res["points"],
                })

        dest_ocr = os.path.join("./app/kie/upload/ocr_result",
               "ocr_Result.json")


        with open(
               dest_ocr,
               "w") as dest_ocrw:
               dest_ocrw.write(json.dumps(ocr_res, ensure_ascii=False))

    else:
        ocr_proprietario=True
        #unico modo per prender ocr result senno
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(
            use_angle_cls=False,
            show_log=False,
            rec_model_dir=None,
            det_model_dir=None, 
            use_gpu=False)

        contents = await file.read()
        file0 = await file.seek(0) 
        os.makedirs("./app/kie/upload", exist_ok=True)
        dest = os.path.join("./app/kie/upload", file.filename)
        s3.put_object(Bucket=cfg.BUCKET_NAME, Key="public/"+cfg.BUCKET_FEEDBACK+file.filename, Body=contents)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer) 


        response_paddle = ocr_engine.ocr(dest,cls=False)


        os.makedirs("./app/kie/upload/ocr_result", exist_ok=True)
        response=formatta_annotation(response_paddle)
        ocr_res=[]
        for res in response:
            ocr_res.append({
                "transcription": res["text"],
                "points":res["points"],
                })

        dest_ocr = os.path.join("./app/kie/upload/ocr_result",
               "ocr_Result_Paddle.json")


        with open(
               dest_ocr,
               "w") as dest_ocrw:
               dest_ocrw.write(json.dumps(ocr_res, ensure_ascii=True))


    ser_re_engine = RePredictor.Inferenza_re()

    #qui va cambiamento infernza su json
    if ocr_proprietario== True:

        with open (dest_ocr,"rb") as f:
            infer_imgs_ocr=f.readlines()
    else:
        infer_imgs_ocr=get_image_file_list(dest)

    base_name=cfg.BUCKET_FEEDBACK
    items={
            "dataset": cfg.DATASET_FEEDBACK_ID,
            "file": {
                "bucket": cfg.BUCKET_NAME,
                "region": "eu-west-1",
                "key":base_name+file.filename,
            },
            "annotations": response,
            "entities":[],
            "relations":[],
            "isInTest":False,
    }
    os.makedirs("./app/save_re/json", exist_ok=True)
    with open(
            os.path.join("./app/save_re/json",
                         "infer_results_"+os.path.splitext(file.filename)[0] +".json"),
            "w",
            encoding='utf-8') as fout:
        for idx, info in enumerate(infer_imgs_ocr):
            if ocr_proprietario== True:
                data_line = info.decode('utf-8')
                substr = data_line.strip("\n").split("\t")
                img_path=dest
                data = {'img_path': dest, 'label': substr[0]}
            else:
                img_path = info
                data = {'img_path': img_path}

            save_img_path = os.path.join(
                "./app/save_re",
                os.path.splitext(os.path.basename(img_path))[0] + "_ser_re.jpg")

            print("~")
            result = ser_re_engine(data)
            result = result[0]
            newresult=dbjson(items,result)

            fout.write(json.dumps(
                newresult, ensure_ascii=False) + "\n")
            await push_item(newresult)


            #img_res = draw_re_results(img_path, result,font_path="./app/kie/simfang.ttf")
            #cv2.imwrite(save_img_path, img_res)



    os.remove(dest)#this remove the image saved
    
    #if ocr_proprietario == True :
        #os.remove(dest_ocr)




#@router.post("/item_push")
async def push_item_test(file: UploadFile=File(...)):
    contents = await file.read()
    s3.put_object(Bucket=cfg.BUCKET_NAME, Key=cfg.BUCKET_FEEDBACK+file.filename, Body=contents)
    print("~")


async def push_item(item):
    tmp= await add_item(item)
    dt=await retrieve_dataset(tmp["dataset"])
    dt["items"].append(str(tmp["_id"]))
    await update_dataset(tmp["dataset"],dt)






def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')



@router.post("/inferenza_paddlehub")
async def inferenzaHub(file: UploadFile=File(...)):
        base_name=cfg.BUCKET_FEEDBACK

    #unico modo per prender ocr result senno
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(
            use_angle_cls=False,
            show_log=False,
            rec_model_dir=None,
            det_model_dir=None, 
            use_gpu=False)

        contents = await file.read()
        file0 = await file.seek(0) 
        os.makedirs("./app/kie/upload", exist_ok=True)
        dest = os.path.join("./app/kie/upload", file.filename)
        s3.put_object(Bucket=cfg.BUCKET_NAME, Key="public/"+cfg.BUCKET_FEEDBACK+file.filename, Body=contents)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer) 


        response_paddle = ocr_engine.ocr(dest,cls=False)
        
        os.makedirs("./app/kie/upload/ocr_result", exist_ok=True)
        response=formatta_annotation(response_paddle)
        ocr_res=[]
        for res in response:
            ocr_res.append({
                "transcription": res["text"],
                "points":res["points"],
                })

        items={
            "dataset": cfg.DATASET_FEEDBACK_ID,
            "file": {
                "bucket": cfg.BUCKET_NAME,
                "region": "eu-west-1",
                "key":base_name+file.filename,
            },
            "annotations": response,
            "entities":[],
            "relations":[],
            "isInTest":False,
        }

        image_file_list = get_image_file_list(dest)

        ####parte nuova
        is_visualize = False
        headers = {"Content-type": "application/json"}
        with open(
            os.path.join("./app/save_re/json",
                         "infer_results_HUB_"+os.path.splitext(file.filename)[0] +".json"),
            "w",
            encoding='utf-8') as fout:
                for image in image_file_list:
                    img=open(image,"rb").read()
                    # seed http request
                    data = {'images': [cv2_to_base64(img)]}
                    print("~sending request~")
                    r = requests.post(
                        url="http://127.0.0.1:8872/predict/kie_ser_re", headers=headers, data=json.dumps(data))

                    res = r.json()["results"][0][0]
                    newresult=dbjson(items,res)
                    fout.write(json.dumps(
                                newresult, ensure_ascii=False) + "\n")



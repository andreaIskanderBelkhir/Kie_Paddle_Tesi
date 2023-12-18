FROM python:3.8.17

RUN apt-get update && apt-get install libgl1 -y

COPY requirements.txt .


RUN python -m pip install -U pip && python -m pip install -r requirements.txt

COPY app /app

#COPY labe_ops.py usr/local/lib/python3.8/site-packages/paddleocr/ppocr/data/imaug/

#COPY vqa_token_re_layoutlm_postprocess.py usr/local/lib/python3.8/site-packages/paddleocr/ppocr/postprocess/

RUN ls

EXPOSE 80

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
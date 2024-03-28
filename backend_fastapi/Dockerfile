FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

EXPOSE 80

#CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "80"]


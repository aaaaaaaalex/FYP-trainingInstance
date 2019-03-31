FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /app

COPY "./app/." "/app/"

RUN ["apt", "update"]
RUN ["pip", "install", "-r", "requirements.txt"]

CMD ["python", "-u",\
        "app.py",\
        #"--config", "dataset/classes.config",\
        "--checkpoint-dir", "./out/1553722589",\
        "--skip-download"]
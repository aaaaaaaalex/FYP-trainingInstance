FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /app

COPY "./app/." "/app/"

RUN ["apt", "update"]
RUN ["pip", "install", "-r", "requirements.txt"]

CMD ["python", "-u",\
        "trainingInstance.py",\
        #"--config", "dataset/classes.config",\
        "--skip-download"]
docker run\
    --runtime=nvidia\
    -v "$(pwd)/app:/app"\
    -u "$(id -u):$(id -g)"\
    --name gpu-env\
    -it tensorflow/tensorflow:latest-gpu-py3

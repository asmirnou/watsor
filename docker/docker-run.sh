#!/usr/bin/env bash

docker run -t -i \
    --rm \
    --env LOG_LEVEL=info \
    --volume /etc/localtime:/etc/localtime:ro \
    --volume $PWD/config:/etc/watsor:ro \
    --device /dev/bus/usb:/dev/bus/usb \
    --device /dev/dri:/dev/dri \
    --publish 8080:8080 \
    --shm-size=512m \
    --gpus all \
    smirnou/watsor.gpu:latest

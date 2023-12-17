#!/bin/bash

DOCKERFILE=simple.Dockerfile
IMAGE_NAME=simple-image

docker build -t ${IMAGE_NAME} -m 100g -f ${DOCKERFILE} .
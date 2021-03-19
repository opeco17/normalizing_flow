FROM python:3.7.0

RUN mkdir /workspace
WORKDIR /workspace

COPY requirements.txt ./

RUN apt-get update && \
  apt-get upgrade -y && \
  pip install --upgrade pip && \
  pip install -r requirements.txt

COPY ./src/* /workspace/
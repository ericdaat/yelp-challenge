##########################################
# Dockerfile to build sentiment analysis #
# application. Based on debian           #
##########################################

FROM debian:latest

USER root
RUN apt-get update
RUN apt-get install -y \
    python \
    python-dev \
    python-distribute \
    python-pip \
    libpcre3 \
    libpcre3-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran


RUN pip install \
    --upgrade pip \
    uwsgi \
    flask \
    numpy \
    scipy \
    -U scikit-learn

RUN useradd -ms /bin/bash wsgi

USER wsgi
WORKDIR /home/sentiment-classifier/

CMD uwsgi wsgi.ini
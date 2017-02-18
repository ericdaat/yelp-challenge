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
    python-scipy \
    python-numpy \
    libpcre3 \
    libpcre3-dev


RUN pip install \
    --upgrade pip \
    uwsgi \
    flask \
    -U scikit-learn

RUN useradd -ms /bin/bash wsgi

USER wsgi
WORKDIR /home/sentiment-classifier/

CMD uwsgi wsgi.ini
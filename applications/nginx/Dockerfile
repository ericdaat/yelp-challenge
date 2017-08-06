##########################################
# Dockerfile for nginx reverse proxy     #
##########################################

FROM nginx

RUN rm /etc/nginx/conf.d/default.conf
COPY ./sentiment-classifier.conf /etc/nginx/conf.d/
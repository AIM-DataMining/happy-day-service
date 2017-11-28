FROM ubuntu:16.04

MAINTAINER Oliver Fesseler

RUN apt-get update
RUN apt-get install -y python3 python3-pip

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8


COPY / /opt/happyday
WORKDIR /opt/happyday

RUN pip3 install -r requirements.txt
#RUN /bin/bash

EXPOSE 5000

ENV FLASK_APP=happyday/service.py
CMD ["flask", "run", "--host=0.0.0.0"]

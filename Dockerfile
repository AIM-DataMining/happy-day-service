FROM ubuntu:16.04

LABEL maintailer="oliver@fesseler.info"

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-pycurl

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

COPY / /opt/happyday
WORKDIR /opt/happyday

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=happyday/service.py
CMD ["flask", "run", "--host=0.0.0.0"]

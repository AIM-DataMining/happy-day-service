FROM ubuntu:16.04

LABEL maintailer="oliver@fesseler.info"

RUN apt-get update \
    && apt-get install -y python3 python3-pip python3-pycurl curl unzip \
    && apt-get clean

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

COPY happyday/ happyday-service/happyday
COPY test/ happyday-service/test
COPY README.md happyday-service/README.md
COPY requirements.txt happyday-service/requirements.txt
WORKDIR happyday-service/

RUN mkdir models
ADD https://schrolm.de/nextcloud/index.php/s/V75Xfk5Udoxp7kP/download models.zip
RUN unzip models.zip

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=happyday/service.py WEBDAV_HOST=https://schrolm.de WEBDAV_LOGIN=${user} WEBDAV_PWD=${pwd} WEBDAV_ROOTDIR=/owncloud/remote.php/webdav
CMD ["flask", "run", "--host=0.0.0.0"]

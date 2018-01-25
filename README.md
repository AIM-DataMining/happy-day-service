# HappyDay
[![Build Status](https://travis-ci.com/ofesseler/happy-day-service.svg?token=XbzPwgk1rQCmoqtErPQz&branch=master)](https://travis-ci.com/ofesseler/happy-day-service)

The aim of the project is to identify the person's facial expression or mood by means of the portrait and assign a suitable emoji. In doing so, we confine ourselves to smiling at the condition.

A typical scenario is the creation of a photo with the smartphone camera in the Android app created in the project.
The face of the person photographed is then recognized and cut out of the image for data reduction.
The tailored facial expression is sent to a server.
On it runs a service that recognizes expressive facial features and body parts from the image detail and assigns them to defined emojis.
Face recognition is realized with a CNN.
The result of the evaluation is sent back to the mobile device and displayed to the user.
He has the possibility to evaluate the received Emoji and send feedback to the server.
In case of an incorrect classification, the image is used for CNN training to continuously improve it.

# HappyDay Service

This repository contains the backend for HappyDay.

## Start

For better testing and deploying I chose Docker.
To execute the commands listed below, you must change to the project directory.

```
docker build -t happyday-image .

docker run --rm -it -p 5000:5000 --name happyday happyday-image
```

The service is then available at http://0.0.0.0:5000


## Endpoints

### GET `/`
Gibt den String `Hello World!\n` zurück.

### GET `/images/`
lists all images in dir.

### GET `/images/<sentiment>`
lists all images in dir `<sentiment>` only "sad" and "smile" are available

### GET `/self-test/<sentiment>`
tests models with predefined image. `<sentiment>` can be one of:
- sad
- smile
- neutral

### POST `/test/`
receives a file and moves it to webdav destination, and later for testing

### POST `/train/<sentiment>`
receives a file and moves it to webdav destination, and later for training

### POST `/retrain/<sentiment>`
NOT YET IMPLEMENTED, empty endpoint. Returns always `{"ok": true}`
retrains a file for given sentiment (sad or smile)


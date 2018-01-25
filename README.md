# HappyDay
[![Build Status](https://travis-ci.com/ofesseler/happy-day-service.svg?token=XbzPwgk1rQCmoqtErPQz&branch=master)](https://travis-ci.com/ofesseler/happy-day-service)

Ziel des Projektes ist es, anhand des Porträts einer Person den Gesichtsausdruck bzw. die Gemütslage zu identifizieren und ein passendes Emoji zuzuordnen.
Dabei beschränken wir uns zunächst auf den Zustand lächeln.

Ein typisches Szenario ist das Erstellen eines Fotos mit der Smartphone-Kamera in der im Projekt erstellten Android App.
Daraufhin wird das Gesicht der fotografierten Person erkannt und zur Datenreduktion aus dem Bild ausgeschnitten.
Der zugeschnittene Gesichtsausdruck wird zu einem Server gesendet.
Auf diesem läuft ein Service, der aus dem Bildausschnitt aussagekräftige Gesichtszüge und Körperteile erkennt und diese definierten Emojis zuordnet.
Realisiert wird die Gesichtserkennung mit einem CNN.
Das Ergebnis der Auswertung wird an das Mobilgerät zurückgeschickt und dem User angezeigt.
Dieser hat die Möglichkeit das empfangene Emoji zu bewerten und Feedback an den Server zu senden.
Bei einer falschen Klassifizierung wird das Bild zum Training des CNN verwendet um dieses kontinuierlich zu verbessern.

# HappyDay Service

In diesem Repository befindet sich das Backend zu HappyDay.

## Start

Zum besseren Testen und Deployen habe ich mich für Docker entschieden.
Um die unten aufgeführten Kommandos auszuführen muss in das Projektverzeichnis gewechselt werden.

```
docker build -t happyday-image .

docker run --rm -it -p 5000:5000 --name happyday happyday-image
```

Anschließend ist der Service unter `http://0.0.0.0:5000` erreichbar.

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


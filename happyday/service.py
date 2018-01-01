import time
from flask import Flask
from flask import request
from flask import json
import os
import io
import webdav.client as wc

from label_image import label_photo
from self_cnn import SelfCNN

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 Mb limit

# don'T forget trailing slash!
BASE_PATH = "/happy-day/"

webdav_options = {
    'webdav_hostname': "https://schrolm.de/nextcloud/remote.php/webdav/",
    'webdav_login': "dm",
    'webdav_password': "Just Smile!",
    'verbose': False
}

client = wc.Client(webdav_options)

self_cnn = SelfCNN()
self_cnn.load("happyday/runs/1513535046095/model-self-cnn.hdf5")


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/images/')
@app.route('/images/<sentiment>')
def images(sentiment=None):
    dir_list = None
    if sentiment == "sad":
        dir_list = client.list(BASE_PATH + "sad")
    elif sentiment == "smile":
        dir_list = client.list(BASE_PATH + "smile")
    else:
        dir_list = client.list(BASE_PATH)
    return json.dumps({"ok": True, "files": dir_list})


@app.route('/test', methods=['POST'])
def prediction():
    local_path, filename = save_to_disk(request.files['photo'].stream)
    # client.upload_sync(remote_path=BASE_PATH + filename, local_path=local_path + filename)
    result = label_photo(local_path + filename)
    result_self_cnn = self_cnn.predict(local_path + filename)
    remove_from_disk(local_path + filename)
    return json.dumps([result, result_self_cnn])


@app.route('/self-test/<sentiment>', methods=['GET'])
def self_test(sentiment):
    result, result_self_cnn = self_test_eval(sentiment)
    return json.dumps([result, result_self_cnn])


def self_test_eval(sentiment):
    filename = ""
    if sentiment == "sad":
        filename = "sad.jpg"
    elif sentiment == "smile":
        filename = "smile.jpg"
    elif sentiment == "neutral":
        filename = "neutral.jpg"

    if filename == "":
        return None, None

    local_path = "happyday/data/"  # save_to_disk(request.files['photo'].stream)
    result = label_photo(local_path + filename)
    result_self_cnn = self_cnn.predict(local_path + filename)
    return result, result_self_cnn


@app.route('/train/<sentiment>', methods=['POST'])
def upload_file(sentiment=None):
    remote_path = None
    local_path, filename = save_to_disk(request.files['photo'].stream)
    if sentiment == "sad":
        remote_path = BASE_PATH + "sad/" + filename
    elif sentiment == "smile":
        remote_path = BASE_PATH + "smile/" + filename
    elif sentiment == "sleep":
        remote_path = BASE_PATH + "sleep/" + filename
    elif sentiment == "kiss":
        remote_path = BASE_PATH + "kiss/" + filename
    elif sentiment == "neutral":
        remote_path = BASE_PATH + "neutral/" + filename
    elif sentiment == "angry":
        remote_path = BASE_PATH + "angry/" + filename
    elif sentiment == "surprised":
        remote_path = BASE_PATH + "surprised/" + filename
    if remote_path is not None:
        client.upload_sync(remote_path=remote_path, local_path=local_path + filename)
        remove_from_disk(local_path + filename)
        return json.dumps({"ok": True})
    else:
        return json.dumps({"ok": False})


@app.route('/retrain/<sentiment>', methods=["POST"])
def retrain(sentiment):
    if sentiment == "sad":
        pass # TODO call retrain method
    elif sentiment == "smile":
        pass # TODO call retrain method
    return json.dumps({"ok": True})


def save_to_disk(data):
    filename = str(int(time.time())) + ".jpg"
    path = "/tmp/"
    with open(path + filename, 'wb') as f:
        f.write(data.read())
        f.close()
    return path, filename


def remove_from_disk(filename):
    try:
        os.remove(filename)
        return True
    except:
        return False

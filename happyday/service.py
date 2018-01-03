import logging
import time
from flask import Flask
from flask import request
from flask import json
import os
import webdav.client as wc

import happyday.label_image as inception
from happyday.self_cnn import SelfCNN

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 Mb limit

# don'T forget trailing slash!
BASE_PATH = "/happy-day/"

webdav_options = {
    'webdav_hostname': "https://schrolm.de",
    'webdav_login': "dm",
    'webdav_password': "rd92c-wPkPi-TG2ta-wCZfo-fbR8n",
    'webdav_root': "/owncloud/remote.php/webdav",
    'verbose': False
}

client = wc.Client(webdav_options)


def models_available(models, path="models"):
    models_exist = True
    try:
        if not os.path.isdir(path):
            os.mkdir(path)
            models_exist = False
        for model in models:
            model_path = "{0}/{1}".format(path, model)
            if not os.path.exists(model_path):
                models_exist = False
                logging.error("please copy model {0} into folder {1}".format(model, path))
    except Exception as e:
        logging.error(e)
        return False
    return models_exist


if models_available(["model-self-cnn.hdf5", "inception-v3-retrained.pb"]):
    # Self-CNN
    self_cnn = SelfCNN()
    self_cnn.load("models/model-self-cnn.hdf5")

    # Retrained InceptionV3
    inception_graph = inception.load_graph("models/inception-v3-retrained.pb")
    inception_labels = inception.load_labels("models/inception-v3-retrained_labels.txt")

    # Selfmade from eisslec
    # TODO load eiselec model
else:
    exit(1)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/images/')
@app.route('/images/<sentiment>')
def images(sentiment=None):
    if sentiment == "sad":
        dir_list = client.list(BASE_PATH + "sad")
    elif sentiment == "smile":
        dir_list = client.list(BASE_PATH + "smile")
    else:
        dir_list = client.list(BASE_PATH)
    return json.dumps({"ok": True, "files": dir_list})


@app.route('/test', methods=['POST'])
def test():
    local_path, filename = save_to_disk(request.files['photo'].stream)
    result, result_self_cnn = prediction(local_path=local_path, filename=filename)
    remove_from_disk(local_path + filename)
    return json.dumps([result, result_self_cnn])


def prediction(local_path, filename):
    result = inception.label_photo(file_name=local_path + filename, graph=inception_graph, labels=inception_labels)
    result_self_cnn = self_cnn.predict(local_path + filename)
    return result, result_self_cnn


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

    local_path = "test/img/"  # save_to_disk(request.files['photo'].stream)
    return prediction(local_path=local_path, filename=filename)


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
        pass  # TODO call retrain method
    elif sentiment == "smile":
        pass  # TODO call retrain method
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

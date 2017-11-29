import time
from flask import Flask
from flask import request
from flask import json
import webdav.client as wc

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
    local_path, filename = save_to_disk(request.data)
    client.upload_sync(remote_path=BASE_PATH + filename, local_path=local_path + filename)
    return json.dumps({"ok": True})


@app.route('/train/<sentiment>', methods=['POST'])
def upload_file(sentiment=None):
    remote_path = None
    local_path, filename = save_to_disk(request.data)
    if sentiment == "sad":
        remote_path = BASE_PATH + "sad/" + filename
    elif sentiment == "smile":
        remote_path = BASE_PATH + "smile/" + filename
    if remote_path is not None:
        client.upload_sync(remote_path=remote_path, local_path=local_path + filename)
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
    path = "tmp/"
    with open(path + filename, 'wb') as f:
        f.write(data)
        f.close()
    return path, filename

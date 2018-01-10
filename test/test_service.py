import logging

import happyday.service as service


def test_hello_world():
    h = service.hello_world()
    assert h == 'Hello, World!'


def test_self_test():
    assert service.self_test_eval("") == (None, None)
    for sentiment in [ "smile", "sad", "neutral"]:
        for model in service.self_test_eval(sentiment):
            logging.info("info: {}: {}".format(sentiment, model[sentiment]))
            if model[sentiment] <= 0.5:
                logging.error("Model eval failed, bad acc. {}: {}".format(sentiment, model[sentiment]))

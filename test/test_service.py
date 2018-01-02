import happyday.service as service


def test_hello_world():
    h = service.hello_world()
    assert h == 'Hello, World!'


def test_self_test():
    assert service.self_test_eval("") == (None, None)
    for model in service.self_test_eval("sad"):
        assert model["sad"] >= 0.5


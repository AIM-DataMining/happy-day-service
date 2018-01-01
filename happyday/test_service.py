import service


def test_hello_world():
    h = service.hello_world()
    assert h == 'Hello, World!'


def test_self_test():
    st = service.self_test_eval("")
    assert st == []
    assert service.self_test_eval("sad")[0] >= 0.5


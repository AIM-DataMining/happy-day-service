import pytest
from happyday.data_processing import DataPrep


path = "test/img/"
moods = ["sad", "smile", "neutral"]
dp = DataPrep()


def test_get_pics_count():
    assert dp.get_pics_count(path, moods) == [9, 10, 6]


def test_get_image_paths_from_mood():
    assert len(dp.get_image_paths_from_mood(path, moods[0])) == 9
    assert dp.get_image_paths_from_mood(path, moods[2])[0] == "1512257872.jpg"


def test_get_random_images():
    assert len(dp.get_random_images(3, path=path, mood=moods[0])) == 3


def test_create_self_cnn_test_set():
    images = dp.get_random_images(5, path=path, mood=moods[0])
    test, train = dp.create_self_cnn_test_sets(images=images)
    assert not test.equals(train)


def test_copy_images():
    _mood = moods[0]
    assert dp.copy_images(images=dp.get_random_images(3, path, _mood),
                          mood=_mood,
                          clazz="test",
                          src_path=path,
                          dest_path="/tmp/happy-day/"
                          )

    assert not dp.copy_images(images=dp.get_random_images(3, path, _mood),
                              mood=_mood,
                              clazz="test",
                              src_path="12w12ew23ei87hg342r",
                              dest_path="/tmp/happy-day/"
                              )

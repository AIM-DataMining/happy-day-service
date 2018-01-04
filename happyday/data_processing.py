import logging
import os
import shutil

import pandas as pd


class DataPrep:
    def get_pics_count(self, path, moods):
        a = []
        for mood in moods:
            a.append(len(os.listdir(path + mood)))
        return a

    def get_image_paths_from_mood(self, path, mood):
        return os.listdir(path + mood)

    def get_random_images(self, count, path, mood):
        random_images = pd.Series(self.get_image_paths_from_mood(path, mood))
        return random_images.sample(n=count).reset_index()

    def create_self_cnn_test_sets(self, images, frac=0.4):
        test_ids = pd.Series(images.index).sample(frac=frac).tolist()
        train_images = images.drop(test_ids)
        test_images = images.iloc[test_ids]
        return test_images, train_images

    def copy_images(self, images, mood, clazz, src_path, dest_path):
        if not os.path.isdir(src_path):
            logging.error("src_path does not exist or isn't a directory")
            return False

        os.makedirs(dest_path + clazz + "/" + mood, exist_ok=True)

        for index, image in images.iterrows():
            image_src = src_path + mood + "/" + image.values[1]
            image_dest = dest_path + clazz + "/" + mood + "/" + image.values[1]
            shutil.copy(image_src, image_dest)
        return True


if __name__ == "__main__":
    _path = "/home/oli/schrolmcloud/Studium/DataMining/happy-day/"
    _moods = ["smile", "sad", "neutral"]
    _clazzes = ["test", "train"]
    dp = DataPrep()
    count = min(dp.get_pics_count(_path, _moods))

    for _mood in _moods:
        _images = dp.get_random_images(count, _path, _mood)
        image_sets = dp.create_self_cnn_test_sets(_images, frac=0.2)
        for i, a in enumerate(image_sets):
            dp.copy_images(images=image_sets[i],
                           mood=_mood,
                           clazz=_clazzes[i],
                           src_path=_path,
                           dest_path="/tmp/happy-day/")

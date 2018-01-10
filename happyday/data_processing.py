import logging
import os
import shutil
from PIL import Image
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

    def copy_images(self, images, mood, clazz, src_path, dest_path, downsample_size=None):
        if not os.path.isdir(src_path):
            logging.error("src_path {} does not exist or isn't a directory".format(src_path))
            return False

        os.makedirs(dest_path + clazz + "/" + mood, exist_ok=True)

        for index, image in images.iterrows():
            image_src = src_path + mood + "/" + image.values[1]
            image_dest = dest_path + clazz + "/" + mood + "/" + image.values[1]
            if downsample_size is not None:
                self.downsample(image_src, image_dest, downsample_size)
            else:
                shutil.copy(image_src, image_dest)
        return True

    def downsample(self, image_src, image_dest, size):
        img = Image.open(image_src)
        img.thumbnail(size, Image.ANTIALIAS)
        img.save(image_dest)

    def asd(self, moods, count, src_path, dst_path):
        for _mood in moods:
            logging.info("copying mood {} to {}".format(_mood, dst_path))
            images = dp.get_random_images(count, src_path, _mood)
            image_sets = dp.create_self_cnn_test_sets(images, frac=0.2)
            for i, a in enumerate(image_sets):
                dp.copy_images(images=image_sets[i],
                               mood=_mood,
                               clazz=_clazzes[i],
                               src_path=src_path,
                               dest_path=dst_path,
                               downsample_size=(256, 256))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source", help="Data source path directory")
    parser.add_argument("-d", "--destination", help="Data destination path directory")

    args = parser.parse_args()
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    _src_path = args.source  # "/home/oli/schrolmcloud/Studium/DataMining/happy-day/"
    _dst_path = args.destination  # "/tmp/happy-day/"

    _moods = ["smile", "sad", "neutral"]
    _clazzes = ["test", "train"]
    dp = DataPrep()
    _count = min(dp.get_pics_count(_src_path, _moods))
    dp.asd(_moods, _count, _src_path, _dst_path)


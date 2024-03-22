#!/usr/bin/env python

import shutil
import fnmatch
import random as rd
import sys
import os


def get_dir_name(path):
    path_split = path.rsplit('/')
    if path_split[-1] == "":
        return path.rsplit('/')[-2]
    else:
        return path.rsplit('/')[-1]


def create_balanced_dir(cnt_dir, max_d, dest, augm_dir):
    """
    For each folder in augm_dir (got names from cnt_dir,
    select randomly an image and copy it into dest_dir)
    """
    for k in cnt_dir.keys():
        cnt = cnt_dir[k]
        print(f"path: {k}, count: {cnt}")
        set_img = set()
        dir_name = get_dir_name(k)
        print(f"directory name: {dir_name}")

        src = augm_dir + dir_name
        print(f"src: {src}")
        if not os.path.isdir(k):
            os.makedirs(k)
        while cnt < max_d:
            tmp = rd.choice(os.listdir(src))
            if os.path.isfile(src + '/' + tmp) and tmp not in set_img:
                set_img.add(tmp)
                shutil.copy2(src + '/' + tmp, k)
                cnt += 1

        print(f"Final count: {cnt}")


if __name__ == "__main__":
    try:

        assert len(sys.argv) >= 2, "Missing argument. Try again."
        src_dir = sys.argv[1]
        if src_dir[-1] != "/":
            src_dir += "/"
        dest_dir = "../augmented_directory/"

        subdirs = [x[0] for x in os.walk(dest_dir)]

        cnt_dir_ = {}
        for dir_ in subdirs[1:]:
            assert os.path.isdir(
                dir_), "subdir should be a valid directory path"
            cnt_dir_[dir_] = len(fnmatch.filter(os.listdir(dir_), '*.JPG'))

        print(cnt_dir_)

        create_balanced_dir(cnt_dir_, 1715, dest_dir, src_dir)

    except Exception as e:
        print(e)

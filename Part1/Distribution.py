import glob
import os
import sys
import matplotlib.pyplot as plt


def count_files(dir_path):
    """ Count files of a directory"""
    cnt = 0
    for path in os.scandir(dir_path):
        if path.is_file():
            cnt += 1
    print(cnt, 'images')
    return cnt


def get_dir_count(subdirs):
    """ Count files into subdirectories of dir_path"""
    subdirs_cnt = {}
    for d in subdirs:
        dir_name = d.rsplit('/', 1)[1]
        print(dir_name, end=": ")
        subdirs_cnt[dir_name] = count_files(d)
    return dict(sorted(subdirs_cnt.items(), key=lambda x: x[1], reverse=True))


def pie_chart(dir_cnt):
    """
    Draw a pie chart from a dictionnary with 4 string keys
    and numerical values)
    """
    plt.pie([int(v) for v in dir_cnt.values()],
            labels=[k for k in dir_cnt.keys()],
            autopct='%0.1f%%',
            colors=['r', 'b', 'g', 'orange'])


def bar_chart(dir_cnt):
    """
    Draw a bar chart from a dictionary with 4 string keys and numerical values)
    """
    plt.bar(range(len(dir_cnt)),
            dir_cnt.values(),
            tick_label=list(dir_cnt.keys()),
            color=['r', 'b', 'g', 'orange'])


if __name__ == "__main__":
    try:
        sys_argv_length = len(sys.argv)
        if len(sys.argv) >= 2:

            path = sys.argv[1]
            if path[-1] != '/':
                path = path + '/'

            subdirs = glob.glob(path+"*")
            subdirs_cnt = get_dir_count(subdirs)
            print(subdirs_cnt)

            pie_chart(subdirs_cnt)
            plt.figure()
            bar_chart(subdirs_cnt)

            plt.show()

        else:
            print("Missing argument. Try again.")
            exit()

    except Exception as e:
        print(e)

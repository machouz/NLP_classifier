# This file provides ass1 which you may or may not find helpful.
# Use it if you want, or ignore it.
import os
import zipfile

import numpy as np

DIC_SEPARATOR = " <=> "


def read_data(fname):
    data = []
    for line in open(fname):
        arr = ["***/STR", "***/STR"] + line[
            :-1
        ].split()  # Add *** with STR tag at the start and remove the "\n" at EOL
        data.append(
            map(lambda x: x.rsplit("/", 1), arr)
        )  # Split between word and tag by the last occurrence of "/"
    return data


def list_to_file(fname, data):
    np.savetxt(fname, data, fmt="%s", delimiter="\n")


def file_to_list(fname):
    return np.loadtxt(fname, delimiter="\n")


def load_labels(fname):
    return np.loadtxt(fname, dtype=np.int16, delimiter="\n")


def map_save_label(labels, labels_id, label_vec_file):
    with open(label_vec_file, "w") as file:
        for label in labels:
            file.write("{}\n".format(labels_id[label]))


def dic_to_file(dic, fname):
    data = []
    for key, label in dic.items():
        data.append(f"{key}{DIC_SEPARATOR}{label}")
    list_to_file(fname, data)


def file_to_dic(fname):
    data = {}
    for line in open(fname):
        key, label = line.rsplit(DIC_SEPARATOR)
        data[key] = int(label)
    return data


def max_nested_dic(nested_dic):
    maxi = -np.inf
    prev = ""
    current = ""
    for key1, val1 in nested_dic.items():
        for key2, val2 in val1.items():
            if val2 > maxi:
                maxi = val2
                prev = key1
                current = key2
    return prev, current


def features_to_file(texts_features, path):
    features_string = []
    for features in texts_features:
        line = ""
        for key, value in features:
            line += "{}={}\t".format(key, value)
        features_string.append(line)
    list_to_file(path, features_string)


def zipFile(path):
    filename = os.path.basename(path)
    directory_path = os.path.dirname(path)
    zip_name = path + ".zip"
    zip_path = os.path.join(directory_path, zip_name)
    zipfile.ZipFile(zip_name, mode="w").write(path, arcname=filename)


def get_dict_from_list_file(path, key=True):
    with open(path) as file:
        elements = file.read().splitlines()
        count = {element: key for element in elements}
    return count

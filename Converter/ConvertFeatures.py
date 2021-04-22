import re

from NLP_classifier import globals
from NLP_classifier.Utils.utils import dic_to_file, list_to_file, file_to_dic

float_regex = re.compile(r"^-?\d+(?:\.\d+)?$")


def check_int(s):
    if s[0] in ("-", "+"):
        return s[1:].isdigit()
    return s.isdigit()


def check_float(s):
    return float_regex.match(s) is not None


def convert_features_to_id_and_save(feature_list, feature_map_file):
    features_id = {}
    i = 0
    for feature in feature_list:
        if feature not in features_id:
            features_id[feature] = i
            i += 1
    dic_to_file(features_id, feature_map_file)
    return features_id


def get_features_id(features_file, feature_map_file, feature_vecs_file):
    data = []
    features_id = file_to_dic(feature_map_file)

    for line in open(features_file):
        stripped_line = line.rstrip()
        if stripped_line:
            splitted_line = stripped_line.split(globals.SEPARTOR)
            features = [get_id_value(feature, features_id) for feature in splitted_line]
        else:
            features = []
        features = map(lambda x: "{}:{}".format(*x), sorted(features))
        data.append("-1 " + " ".join(features))

    list_to_file(feature_vecs_file, data)
    return data


def get_id_value(feature, features_id):
    splitted = feature.split(globals.EQUALITY)
    assert len(splitted) == 2
    if check_float(splitted[1]):
        feature, value = splitted[0], splitted[1]
    else:
        value = 1

    return features_id[feature], value

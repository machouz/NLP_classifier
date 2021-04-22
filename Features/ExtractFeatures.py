from NLP_classifier import globals
from NLP_classifier.Utils import features_utils, utils


def word_gram(texts, features_path, frquencies=True):
    grams = get_grams_file(features_path)
    selected = utils.get_dict_from_list_file(features_path)
    grams_texts = [
        features_utils.get_selected_word_gram(text, grams, selected, frquencies)
        for text in texts
    ]

    return grams_texts


def char_gram(texts, features_path, frquencies=True):
    grams = get_grams_file(features_path)
    selected = utils.get_dict_from_list_file(features_path)
    grams_texts = [
        features_utils.get_selected_char_gram(text, grams, selected, frquencies)
        for text in texts
    ]

    return grams_texts


def extract(texts, functions, save_features_path):
    features_extracted = [extract_function(texts) for extract_function in functions]
    features_extracted = list(zip(*features_extracted))
    features_extracted_dic = [
        {key: value for dic in row for key, value in dic.items()}
        for row in features_extracted
    ]
    save_list_of_features_by_entries(features_extracted_dic, save_features_path)


def get_grams_file(features_path):
    grams = {}
    for item in open(features_path):
        gram = len(item.rsplit())
        if gram not in grams:
            grams[gram] = True

    return list(grams.keys())


def get_feature_equality(arr):
    return f"{arr[0]}{globals.EQUALITY}{arr[1]:.15f}"



def save_list_of_features_by_entries(entries, path):
    with open(path, "w") as file:
        for entry in entries:
            file.write(
                globals.SEPARTOR.join(
                    [get_feature_equality(item) for item in entry.items()]
                    ) + "\n"
            )

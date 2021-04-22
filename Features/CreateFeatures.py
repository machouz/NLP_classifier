import re
from collections import Counter

from NLP_classifier.Utils.features_utils import text_char_gram, text_word_gram

word_regex = re.compile("word_(\d+)gram_(\d+)")
char_regex = re.compile("char_(\d+)gram_(\d+)")


def get_occurences(path, normalization=None):
    with open(path) as file:
        lines = file.readlines()
        result = list(map(str.rstrip, lines))
        if normalization:
            result = list(map(normalization, result))
        return result


def get_function_from_word_conf(configurations):
    def get_results(texts):
        results = []
        if configurations:
            for configuration in configurations:
                for gram, size in word_regex.findall(configuration):
                    results.extend(word_gram(texts, gram=int(gram), n_best=int(size)))
            return list(set(results))
        else:
            return []

    return get_results


def get_function_from_char_conf(configurations):
    def get_results(texts):
        results = []
        if configurations:
            for configuration in configurations:
                for gram, size in char_regex.findall(configuration):
                    results.extend(char_gram(texts, gram=int(gram), n_best=int(size)))
            return list(set(results))
        else:
            return []

    return get_results


def word_gram(texts, gram=1, n_best=100):
    grams_list = []
    for i, text in enumerate(texts):
        grams_list.extend(text_word_gram(text, gram))
    return [i[0] for i in Counter(grams_list).most_common(n_best)]


def char_gram(texts, gram=1, n_best=100):
    grams_list = []
    for text in texts:
        grams_list.extend(text_char_gram(text, gram))
    return [i[0] for i in Counter(grams_list).most_common(n_best)]

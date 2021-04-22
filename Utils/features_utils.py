import itertools
from collections import Counter

features = {
    "number_of_characters": lambda series: list(
        series.apply(lambda x: {'number_of_characters': get_number_of_characters(x)})),
    "number_of_words": lambda series: list(series.apply(lambda x: {'number_of_words': get_number_of_words(x)})),
}


def get_number_of_words(text):
    return len(text.split())


def get_number_of_characters(text):
    return len(text)


def get_func(str_list=None):
    if not str_list:
        return []
    return [features[element] for element in str_list]


def get_selected_word_gram(text, grams, count, frequencies=True):
    grams_list = text_word_grams_list(text, grams)
    total = len(grams_list) if frequencies else 1
    grams_counter = Counter(grams_list)
    selected_grams = {item: grams_counter[item] / total for item in grams_counter.keys() if item in count}
    return selected_grams


def get_selected_char_gram(text, grams, count, frequencies=True):
    grams_list = text_char_grams_list(text, grams)
    total = len(grams_list) if frequencies else 1
    grams_counter = Counter(grams_list)
    selected_grams = {item: grams_counter[item] / total for item in grams_counter.keys() if item in count}
    return selected_grams


def text_word_gram(text, gram):
    splited_text = text.rsplit()
    return [' '.join(splited_text[i:i + gram]) for i in range(0, len(splited_text) - gram + 1)]


def text_word_grams_list(text, grams):
    text = text.rstrip()
    return list(itertools.chain(*[text_word_gram(text, gram) for gram in grams]))


def text_char_gram(text, gram):
    words = text_word_gram(text, gram=1)
    return [' '.join(word[i:i + gram]) for word in words for i in range(0, len(word) - gram + 1)]


def text_char_grams_list(text, grams):
    text = text.rstrip()
    return list(itertools.chain(*[text_char_gram(text, gram) for gram in grams]))

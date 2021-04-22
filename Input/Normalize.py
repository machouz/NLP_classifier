import re

START_TOKEN = '<SENTENCE> '
END_TOKEN = ' </SENTENCE>'


class Turkish:
    @staticmethod
    def normalize(text):
        return START_TOKEN + text.lower() + END_TOKEN


class Danish:
    @staticmethod
    def normalize(text):
        return START_TOKEN + text.lower() + END_TOKEN


class Greek:
    @staticmethod
    def normalize(text):
        return START_TOKEN + text.lower() + END_TOKEN


class Arabic:
    @staticmethod
    def deNoise(text):
        noise = re.compile(""" ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
        text = re.sub(noise, '', text)
        return text

    @staticmethod
    def normalizeArabic(text):
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        return text

    @staticmethod
    def normalize(text):
        text = text.lower()
        text = Arabic.deNoise(text)
        text = Arabic.normalizeArabic(text)
        return START_TOKEN + text.lower() + END_TOKEN


normalize_language = {
    'greek': Greek.normalize,
    'arabic': Arabic.normalize,
    'turkish': Turkish.normalize,
    'danish': Danish.normalize
}

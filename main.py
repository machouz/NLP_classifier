from sklearn.model_selection import train_test_split

from NLP_classifier import Model
from NLP_classifier.Converter import ConvertFeatures, ConvertLabels
from NLP_classifier.Features import ExtractFeatures, CreateFeatures
from NLP_classifier.Input import ReadFile
from NLP_classifier.Input.Normalize import normalize_language
from NLP_classifier.Output import CreateEnv, SaveResult
from NLP_classifier.Utils import utils, model_utils, features_utils
from NLP_classifier.Utils.utils import zipFile
from NLP_classifier.globals import Globals


class Main:
    normalization = None
    test = None

    def __init__(
        self,
        train_path,
        features,
        features_creation,
        model_names,
        result,
        output,
        dev_path=None,
        test_path=None,
        words_occurence_features=None,
        chars_occurence_features=None,
        normalize=False,
        verbose=False
    ):

        self.paths = CreateEnv.create(output, model_names, result)
        self.train = ReadFile.read_file(train_path)

        self.verbose = verbose

        if not dev_path:
            self.train, self.dev = train_test_split(self.train, test_size=0.2)
        else:
            self.dev = ReadFile.read_file(dev_path)
        self.test = (
            ReadFile.read_file(test_path, names=["id", "text"]) if test_path else None
        )
        if normalize:
            self.normalization = normalize_language[normalize]
            self.train.text = self.train.text.apply(self.normalization)
            self.test.text = self.test.text.apply(self.normalization)

            if self.test is not None:
                self.test.text = self.test.text.apply(self.normalization)
        self.model_names = model_names
        self.models = model_utils.get_models(model_names)

        # Features creator
        self.features_creation = features_creation
        self.word_feature_creation = CreateFeatures.get_function_from_word_conf(
            features_creation
        )
        self.words_occurence_features = words_occurence_features

        self.char_feature_creation = CreateFeatures.get_function_from_char_conf(
            features_creation
        )
        self.chars_occurence_features = chars_occurence_features

        self.features = features if features else []
        self.features_functions = [
            lambda x: ExtractFeatures.word_gram(x, self.paths["word_features_path"]),
            lambda x: ExtractFeatures.char_gram(x, self.paths["char_features_path"]),
            *features_utils.get_func(self.features),
        ]

    def create_features_labels(self):
        # Created features
        word_features_list = self.word_feature_creation(self.train.text)
        if self.words_occurence_features:
            words_occurence_features = []
            for path in self.words_occurence_features:
                words_occurence_features.extend(
                    CreateFeatures.get_occurences(path, self.normalization)
                )

            word_features_list += words_occurence_features
        utils.list_to_file(self.paths["word_features_path"], sorted(word_features_list))

        char_features_list = self.char_feature_creation(self.train.text)
        if self.chars_occurence_features:
            chars_occurence_features = []
            for path in self.chars_occurence_features:
                chars_occurence_features.extend(CreateFeatures.get_occurences(path))

            char_features_list += chars_occurence_features
        utils.list_to_file(self.paths["char_features_path"], sorted(char_features_list))

        all_features = word_features_list + char_features_list + self.features
        Globals.FEATURES = len(all_features)
        if self.verbose:
            print("Features created")
        # Convert features to ids
        ConvertFeatures.convert_features_to_id_and_save(
            all_features, feature_map_file=self.paths["features_map_path"]
        )
        if self.verbose:
            print("Features mapped")

        # Get Labels
        labels = list(set(self.train.label))

        # Convert labels to ids
        ConvertLabels.convert_labels_to_id_and_save(
            labels, label_map_file=self.paths["labels_map_path"]
        )
        if self.verbose:
            print("Labels mapped")

    def extract_convert_train_feature(self,):
        # Extract and convert train features and labels
        ExtractFeatures.extract(
            self.train.text,
            functions=self.features_functions,
            save_features_path=self.paths["train_extracted_features"],
        )
        if self.verbose:
            print("Train features extracted")

        ConvertFeatures.get_features_id(
            self.paths["train_extracted_features"],
            feature_map_file=self.paths["features_map_path"],
            feature_vecs_file=self.paths["train_vec_extracted_features"],
        )
        ConvertLabels.get_labels_id(
            labels=self.train.label,
            label_map_file=self.paths["labels_map_path"],
            label_vec_file=self.paths["train_label"],
        )
        if self.verbose:
            print("Train features converted")

    def extract_convert_dev_labels(self,):
        # Extract and convert dev features and labels
        ExtractFeatures.extract(
            self.dev.text,
            functions=self.features_functions,
            save_features_path=self.paths["dev_extracted_features"],
        )
        if self.verbose:
            print("Dev features extracted")
        ConvertFeatures.get_features_id(
            self.paths["dev_extracted_features"],
            feature_map_file=self.paths["features_map_path"],
            feature_vecs_file=self.paths["dev_vec_extracted_features"],
        )
        ConvertLabels.get_labels_id(
            labels=self.dev.label,
            label_map_file=self.paths["labels_map_path"],
            label_vec_file=self.paths["dev_label"],
        )
        if self.verbose:
            print("Dev features converted")

    def extract_convert_test(self,):
        # Extract and convert dev features and labels
        ExtractFeatures.extract(
            self.test.text,
            functions=self.features_functions,
            save_features_path=self.paths["test_extracted_features"],
        )
        if self.verbose:
            print("Test features extracted")

        ConvertFeatures.get_features_id(
            self.paths["test_extracted_features"],
            feature_map_file=self.paths["features_map_path"],
            feature_vecs_file=self.paths["test_vec_extracted_features"],
        )
        if self.verbose:
            print("Test features converted")

    def do_all(self):

        self.create_features_labels()
        self.extract_convert_train_feature()
        self.extract_convert_dev_labels()
        if self.test is not None:
            self.extract_convert_test()

        for i, model in enumerate(self.models):
            print(model)
            # Use features to train
            Model.train_model(
                model=self.models[i],
                feature_vecs_file=self.paths["train_vec_extracted_features"],
                label_file=self.paths["train_label"],
                model_path=self.paths["models"][i],
            )
            if self.verbose:
                print("Model trained")

            # Eval
            f1 = Model.eval_model(
                model_path=self.paths["models"][i],
                feature_vecs_file=self.paths["dev_vec_extracted_features"],
                label_file=self.paths["dev_label"],
            )

            SaveResult.toCSV(
                self.model_names[i],
                features=self.features_creation,
                f1=f1,
                path=self.paths["csv_result"],
                added_features=[
                    self.words_occurence_features,
                    self.chars_occurence_features,
                ],
            )

            if self.test is not None:
                Model.predict(
                    ids=self.test.id,
                    model_path=self.paths["models"][i],
                    feature_vecs_file=self.paths["test_vec_extracted_features"],
                    label_map_file=self.paths["labels_map_path"],
                    pred_file=self.paths["test_predictions"][i],
                    sep=",",
                )
                zipFile(self.paths["test_predictions"][i])


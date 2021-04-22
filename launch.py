import argparse
import os, sys
import random

import numpy as np
from tqdm import tqdm

from NLP_classifier.Utils import model_utils, features_utils
from NLP_classifier.Utils.argparse_utils import LoadFromFile, args_to_file
from NLP_classifier.main import Main

parser = argparse.ArgumentParser(description="Text classifier")
parser.add_argument(
    "-tr", "--train_path", help="Path to the train file", required=False
)
parser.add_argument("-d", "--dev_path", help="Path to the dev file", required=False)
parser.add_argument("-tt", "--test_path", help="Path to the test file", required=False)
parser.add_argument("-n", "--normalize", help="Langage to normalize", required=False)
parser.add_argument(
    "-f",
    "--features",
    nargs="+",
    choices=features_utils.features.keys(),
    help="Features to use",
    required=False,
)
parser.add_argument(
    "-fc",
    "--features_creation",
    nargs="+",
    help="Features to create",
    action="append",
    required=False,
)
parser.add_argument(
    "-wo",
    "--words_occurence_features",
    required=False,
    action="append",
    help="Path of list of word that occurence is a features to create",
)
parser.add_argument(
    "-co",
    "--chars_occurence_features",
    required=False,
    action="append",
    help="Path of list of char that occurence is a features to create",
)
parser.add_argument(
    "-mn",
    "--model_names",
    help="Model to develop",
    choices=model_utils.methods.keys(),
    required=False,
    action="append",
)
# parser.add_argument('-mp', '--model_path', help='Path to trained model to use', required=False, action='append')
parser.add_argument("--file", type=open, action=LoadFromFile)
parser.add_argument("-o", "--output", help="Directory output", required=False)

random.seed(9001)
np.random.seed(9001)

if __name__ == "__main__":
    args = parser.parse_args()
    for i, features_creation in enumerate(tqdm(args.features_creation)):
        features_to_create = " ".join(features_creation)
        output = os.path.join(args.output, features_to_create)
        print(f"{i + 1}/{len(args.features_creation)}: Running for features {features_to_create}")
        fm = Main(
            train_path=args.train_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            normalize=args.normalize,
            features=args.features,
            features_creation=features_creation,
            model_names=args.model_names,
            words_occurence_features=args.words_occurence_features,
            chars_occurence_features=args.chars_occurence_features,
            output=output,
            result=args.output,
        )
        args_to_file(os.path.join(output, "description.txt"), args)
        fm.do_all()

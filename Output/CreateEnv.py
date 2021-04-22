import os
from pathlib import Path


def create(output_path, model_names, result=None):
    features_path = os.path.join(output_path, "features")
    extracted_features = os.path.join(output_path, "extracted_features")

    word_features_path = os.path.join(features_path, "word.txt")
    char_features_path = os.path.join(features_path, "character.txt")

    features_map_path = os.path.join(output_path, "features_map.txt")
    labels_map_path = os.path.join(output_path, "labels_map.txt")

    train_extracted_features = os.path.join(extracted_features, "train.txt")
    train_vec_extracted_features = os.path.join(extracted_features, "train_vec.txt")
    train_label = os.path.join(extracted_features, "train_label.txt")

    dev_extracted_features = os.path.join(extracted_features, "dev.txt")
    dev_vec_extracted_features = os.path.join(extracted_features, "dev_vec.txt")
    dev_label = os.path.join(extracted_features, "dev_label.txt")

    test_extracted_features = os.path.join(extracted_features, "test.txt")
    test_vec_extracted_features = os.path.join(extracted_features, "test_vec.txt")

    Path(features_path).mkdir(parents=True, exist_ok=True)

    Path(extracted_features).mkdir(parents=True, exist_ok=True)
    if not result:
        result = output_path
    csv_result = os.path.join(result, "result.csv")

    test_predictions = []
    model_paths = []
    for model in model_names:
        model_path = os.path.join(output_path, model)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        test_predictions.append(os.path.join(model_path, "test_predictions.csv"))
        model_paths.append(os.path.join(model_path, "model.pkl"))

    return {
        "features_path": features_path,
        "word_features_path": word_features_path,
        "char_features_path": char_features_path,

        'features_map_path': features_map_path,
        'labels_map_path': labels_map_path,

        'extracted_features': extracted_features,

        'train_extracted_features': train_extracted_features,
        'train_vec_extracted_features': train_vec_extracted_features,
        'train_label': train_label,

        'dev_extracted_features': dev_extracted_features,
        'dev_vec_extracted_features': dev_vec_extracted_features,
        'dev_label': dev_label,

        'test_extracted_features': test_extracted_features,
        'test_vec_extracted_features': test_vec_extracted_features,

        'test_predictions': test_predictions,
        'models': model_paths,
        'csv_result': csv_result
    }

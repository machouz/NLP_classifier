import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest, chi2

from NLP_classifier.Utils import model_utils
from NLP_classifier.Utils.utils import load_labels, file_to_dic
from NLP_classifier.globals import Globals
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


def selection(X, Y):
    return SelectKBest(chi2, k=2000).fit_transform(X, Y)


def get_X_train_Y_train(feature_vecs_file, label_file):
    X_train, _ = load_svmlight_file(
        feature_vecs_file, n_features=Globals.FEATURES)
    Y_train = load_labels(label_file)

    return X_train, Y_train


def cross_train_eval(model, feature_vecs_file, label_file, model_path, cross_fold=10):
    X_train, Y_train = get_X_train_Y_train(feature_vecs_file, label_file)

    # prepare the cross-validation procedure
    cv = KFold(n_splits=cross_fold, random_state=5, shuffle=True)

    result = cross_validate(model, X_train, Y_train,
                            cv=cv, scoring=['precision', 'recall', 'f1_macro', 'accuracy'])
    print(
        f"Precision: {result['test_precision'].mean()}, Recall: {result['test_recall'].mean()}, F1:{result['test_f1_macro'].mean()}, accuracy:{result['test_accuracy'].mean()}")
    model_utils.save_model(model, model_path)

    return model, result['test_accuracy'].mean()


def train_model(model, feature_vecs_file, label_file, model_path):
    X_train, Y_train = get_X_train_Y_train(feature_vecs_file, label_file)

    # print("SVM light loaded, lenght {}".format(Y_train.shape))

    model.fit(X_train, Y_train)

    model_utils.save_model(model, model_path)

    return model


def predict(ids, model_path, pred_file, feature_vecs_file, label_map_file, sep="\t"):
    model = model_utils.load_model(model_path)
    id_label = file_to_dic(label_map_file)
    label_id = {key: value for value, key in id_label.items()}
    X_train, _ = load_svmlight_file(
        feature_vecs_file, n_features=Globals.FEATURES)
    y_pred = model.predict(X_train)
    with open(pred_file, "w") as file:
        for i, y_pred in zip(ids, y_pred):
            file.write("{}{}{}\n".format(i, sep, label_id[y_pred]))
    return y_pred


def eval_model(model_path, feature_vecs_file, label_file):
    model = model_utils.load_model(model_path)
    X_dev, _ = load_svmlight_file(
        feature_vecs_file, n_features=Globals.FEATURES)
    y_gold = load_labels(label_file)
    y_pred = model.predict(X_dev)

    precision = sklearn.metrics.precision_score(
        y_gold, y_pred, average="macro")
    recall = sklearn.metrics.recall_score(y_gold, y_pred, average="macro")
    f1 = sklearn.metrics.f1_score(y_gold, y_pred, average="macro")
    accuracy = sklearn.metrics.accuracy_score(y_gold, y_pred)

    print(
        f"Precision: {precision}, Recall: {recall}, F1:{f1}, accuracy:{accuracy}")
    return f1

import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

RANDOM_SEED = 5

methods = {
    "svc": LinearSVC(random_state=RANDOM_SEED),
    "balanced_svc": LinearSVC(class_weight='balanced', random_state=RANDOM_SEED),
    "kn": KNeighborsClassifier(),
    "rf": RandomForestClassifier(random_state=RANDOM_SEED),
    "abf": AdaBoostClassifier(random_state=RANDOM_SEED),
    "mlp": MLPClassifier(random_state=RANDOM_SEED),
    "lr": LogisticRegression(random_state=RANDOM_SEED),
    "mnb": MultinomialNB(),
}


def get_models(str_list):
    return [methods[element] for element in str_list]


def save_model(clf, path):
    with open(path, 'wb') as f:
        pickle.dump(clf, f)


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

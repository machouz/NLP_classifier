import pickle

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.mixture import DPGMM
from sklearn.mixture import GMM 
from sklearn.mixture import GaussianMixture
from sklearn.mixture import VBGMM

RANDOM_SEED = 5

methods = {
    "svc": LinearSVC(random_state=RANDOM_SEED),
    "balanced_svc": LinearSVC(class_weight='balanced', random_state=RANDOM_SEED),
    "dt": DecisionTreeClassifier(random_state=RANDOM_SEED),
    "rf": RandomForestClassifier(random_state=RANDOM_SEED),
    "etc": ExtraTreesClassifier(random_state=RANDOM_SEED),
    "mlp": MLPClassifier(random_state=RANDOM_SEED),
    "lr": LogisticRegression(random_state=RANDOM_SEED),
    "lrcv": LogisticRegressionCV(random_state=RANDOM_SEED),
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

model_registry = {
    "rf":{
        "estimator":RandomForestClassifier(random_state=44),
        "param_grid":{
            "classifier__n_estimators": [100, 200, 500, 1000],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5]
        }
    },
    "logreg": {
        "estimator": LogisticRegression(max_iter=1000),
        "param_grid": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ['l2'],
            "classifier__solver": ['lbfgs']
        }
    },
    "svm": {
        "estimator": SVC(),
        "param_grid": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ['linear', 'rbf'],
            "classifier__gamma": ['scale', 'auto']
        }
    },
    "knn": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "classifier__n_neighbors": [3, 5, 7],
            "classifier__weights": ['uniform', 'distance']
        }
    }
    # "mlp":{
    #     "estimator": MLPClassifier(max_iter=2000),
    #     "param_grid": {
    #         "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
    #         "classifier__activation": ["relu", "tanh"],
    #         "classifier__solver": ["adam"],
    #         "classifier__alpha": [0.0001, 0.001],
    #         "classifier__learning_rate":["constant","adaptive"],
    #
    #     }
    # }
}
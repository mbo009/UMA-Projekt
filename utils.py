from sklearn.model_selection import KFold
from eval_techniques import *
import torch


def kfold_cv(model_class, params, X, y, eval_fns):
    evaluations = [[] for _ in range(len(eval_fns))]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
        X_train_cv_tensor = torch.tensor(X_train_cv, dtype=torch.float32)
        y_train_cv_tensor = torch.tensor(y_train_cv, dtype=torch.float32).view(-1, 1)
        X_test_cv_tensor = torch.tensor(X_test_cv, dtype=torch.float32)

        model = model_class(params)
        model.train(X_train_cv_tensor, y_train_cv_tensor)
        predictions = model.predict(X_test_cv_tensor)
        predictions = predictions >= 0.5
        for i, eval_fn in enumerate(eval_fns):
            evaluations[i].append(eval_fn(predictions, y_test_cv))

    return evaluations


def compare_hyperparams(model_class, params, X, y):
    error_rates = [[] for _ in range(len(params))]

    for i, param in enumerate(params):
        error_rates[i].append(np.mean(kfold_cv(model_class, param, X, y, error_rate)))

    return error_rates


def compare_models(model_classes, params, X, y):
    eval_fns = [
        error_rate,
        accuracy,
        true_positive_rate,
        false_positive_rate,
        precision,
        f1_score,
    ]
    output = {}
    for i, model_class in enumerate(model_classes):
        model_evals = {}
        model_eval = kfold_cv(model_class, X, y, eval_fns)
        for j, eval_fn in enumerate(eval_fns):
            model_evals[eval_fn.__name__] = np.mean(model_eval[j])
        output[model_class.__name__] = model_evals

    return output

from sklearn.model_selection import KFold
from eval_techniques import *
import torch
from time import perf_counter


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
        error_rates[i].append(np.mean(kfold_cv(model_class, param, X, y, [error_rate])))

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
        model_eval = kfold_cv(model_class, params[i], X, y, eval_fns)
        for j, eval_fn in enumerate(eval_fns):
            model_evals[eval_fn.__name__] = np.mean(model_eval[j])
        output[model_class.__name__] = model_evals

    return output


def measure_times(model_class, params, X_train, y_train, X_test, n_times=50):
    train_times = []
    predict_times = []
    for param in params:
        tmp_train_times, tmp_predict_times = [], []
        for _ in n_times:
            model = model_class(param[0])
            start_time = perf_counter()
            model.train(X_train, y_train)
            tmp_train_times.append(perf_counter() - start_time)

            start_time = perf_counter()
            model.predict(X_test)
            tmp_predict_times.append(perf_counter() - start_time)
        train_times.append(np.mean(tmp_train_times))
        predict_times.append(np.mean(tmp_predict_times))

    return train_times, predict_times


def plot_roc(tpr, fpr):
    plt.plot(fpr[0], tpr[0], label="MLP", color="green")
    plt.plot(fpr[1], tpr[1], label="KNN", color="red")
    plt.plot(fpr[2], tpr[2], label="RF", color="blue")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("Krzywe ROC")
    plt.grid()
    plt.show()


def plot_roc_curve(model_classes, params, X_train, y_train, X_test, y_test):
    tpr, fpr = [0, 0, 0], [0, 0, 0]
    for i, (model_class, param) in enumerate(zip(model_classes, params)):
        tpr[i], fpr[i] = generate_roc_data(
            model_class, params, X_train, y_train, X_test, y_test
        )
    plot_roc_curve(tpr, fpr)


def generate_roc_data(model_class, param, X_train, y_train, X_test, y_test):
    model = model_class(param)
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    return roc_curve(predictions, y_test)

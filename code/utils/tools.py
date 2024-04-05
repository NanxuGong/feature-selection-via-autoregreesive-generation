import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


def downstream_task_new(data, task_type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0, n_jobs=128)
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0, n_jobs=128)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'det':
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=128)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_jobs=128))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    elif task_type == 'rank':
        pass
    else:
        return -1

# 'RF', 'XGB', 'SVM', 'KNN', 'Ridge'
def downstream_task_by_method(data, task_type, method):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso()
    else:  # dt
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    else:
        return -1

def downstream_task_by_method_no_k_fold(data, task_type, method, init_seed=1):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    init_seed = 1
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso()
    else:  # dt
        init_seed=2
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=init_seed, shuffle=True)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        rae_list = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=init_seed, shuffle=True)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=init_seed, shuffle=True)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    else:
        return -1

def downstream_task_by_method_std(data, task_type, method):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso()
    else:  # dt
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list), np.std(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list), np.std(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list), np.std(f1_list)
    else:
        return -1



def test_task_wo_cv(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            auc_roc_score.append(roc_auc_score(y_test, y_predict, average='weighted'))
            break
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list, rmse_list = [], [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict, squared=True))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
            break
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        recall = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            ras.append(roc_auc_score(y_test, y_predict))
            recall.append(recall_score(y_test, y_predict, average='weighted'))
            break
        return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    elif task == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, maf1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='macro'))
            rec_list.append(recall_score(y_test, y_predict, average='macro'))
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
            maf1_list.append(f1_score(y_test, y_predict, average='macro'))
            break
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    elif task == 'rank':
        pass
    else:
        return -1

def test_task_new(Dg, task='cls', init_seed=0):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=init_seed, shuffle=True)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        pre_list.append(precision_score(y_test, y_predict, average='weighted'))
        rec_list.append(recall_score(y_test, y_predict, average='weighted'))
        f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        auc_roc_score.append(roc_auc_score(y_test, y_predict, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
    elif task == 'reg':
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list, rmse_list = [], [], [], []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=init_seed, shuffle=True)
        reg.fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        mae_list.append(1 - mean_absolute_error(y_test, y_predict))
        mse_list.append(1 - mean_squared_error(y_test, y_predict, squared=True))
        rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
    elif task == 'det':
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        recall = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=init_seed, shuffle=True)
        knn_model.fit(X_train, y_train)
        y_predict = knn_model.predict(X_test)
        map_list.append(average_precision_score(y_test, y_predict))
        f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        ras.append(roc_auc_score(y_test, y_predict))
        recall.append(recall_score(y_test, y_predict, average='weighted'))
        return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    elif task == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, maf1_list = [], [], [], []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=init_seed, shuffle=True)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        pre_list.append(precision_score(y_test, y_predict, average='macro'))
        rec_list.append(recall_score(y_test, y_predict, average='macro'))
        f1_list.append(f1_score(y_test, y_predict, average='micro'))
        maf1_list.append(f1_score(y_test, y_predict, average='macro'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    # if task == 'cls':
    #     clf = RandomForestClassifier(random_state=0)
    #     pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
    #     skf = StratifiedKFold(n_splits=5, random_state=init_seed, shuffle=True)
    #     for train, test in skf.split(X, y):
    #         X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
    #         clf.fit(X_train, y_train)
    #         y_predict = clf.predict(X_test)
    #         pre_list.append(precision_score(y_test, y_predict, average='weighted'))
    #         rec_list.append(recall_score(y_test, y_predict, average='weighted'))
    #         f1_list.append(f1_score(y_test, y_predict, average='weighted'))
    #         auc_roc_score.append(roc_auc_score(y_test, y_predict, average='weighted'))
    #     return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
    # elif task == 'reg':
    #     kf = KFold(n_splits=5, random_state=init_seed, shuffle=True)
    #     reg = RandomForestRegressor(random_state=0)
    #     mae_list, mse_list, rae_list, rmse_list = [], [], [], []
    #     for train, test in kf.split(X):
    #         X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
    #         reg.fit(X_train, y_train)
    #         y_predict = reg.predict(X_test)
    #         mae_list.append(1 - mean_absolute_error(y_test, y_predict))
    #         mse_list.append(1 - mean_squared_error(y_test, y_predict, squared=True))
    #         rae_list.append(1 - relative_absolute_error(y_test, y_predict))
    #         rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
    #     return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
    # elif task == 'det':
    #     kf = KFold(n_splits=5, random_state=init_seed, shuffle=True)
    #     knn_model = KNeighborsClassifier(n_neighbors=5)
    #     map_list = []
    #     f1_list = []
    #     ras = []
    #     recall = []
    #     for train, test in kf.split(X):
    #         X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
    #         knn_model.fit(X_train, y_train)
    #         y_predict = knn_model.predict(X_test)
    #         map_list.append(average_precision_score(y_test, y_predict))
    #         f1_list.append(f1_score(y_test, y_predict, average='weighted'))
    #         ras.append(roc_auc_score(y_test, y_predict))
    #         recall.append(recall_score(y_test, y_predict, average='weighted'))
    #     return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    # elif task == 'mcls':
    #     clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
    #     pre_list, rec_list, f1_list, maf1_list = [], [], [], []
    #     skf = StratifiedKFold(n_splits=5, random_state=init_seed, shuffle=True)
    #     for train, test in skf.split(X, y):
    #         X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
    #         clf.fit(X_train, y_train)
    #         y_predict = clf.predict(X_test)
    #         pre_list.append(precision_score(y_test, y_predict, average='macro'))
    #         rec_list.append(recall_score(y_test, y_predict, average='macro'))
    #         f1_list.append(f1_score(y_test, y_predict, average='micro'))
    #         maf1_list.append(f1_score(y_test, y_predict, average='macro'))
    #     return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    elif task == 'rank':
        pass
    else:
        return -1

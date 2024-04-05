# filter methods
# mRMR. The mRMR [28] firstly ranks features by minimizing feature’s redundancy,
# while maximizing their relevance with the target vector (label vector), and then
# selects the K highest ranking features.
# In the experiments, we make K equal to the number of selected features in MARLFS.
import pickle
# import pyDecision
from pyDecision.algorithm import vikor_method
import warnings
warnings.filterwarnings("ignore")
import mcdm
from genetic_selection import GeneticSelectionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression, Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVR, LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import safe_sqr
from operator import attrgetter

from xgboost import XGBClassifier, XGBRegressor

from utils.logger import info
import torch
from feature_env import FeatureEvaluator, base_path
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE, SelectFromModel
import numpy as np


def get_feature_importances(estimator, getter, transform_func=None, norm_order=1):
    """
    Retrieve and aggregate (ndim > 1)  the feature importances
    from an estimator. Also optionally applies transformation.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator from which we want to get the feature
        importances.

    getter : "auto", str or callable
        An attribute or a callable to get the feature importance. If `"auto"`,
        `estimator` is expected to expose `coef_` or `feature_importances`.

    transform_func : {"norm", "square"}, default=None
        The transform to apply to the feature importances. By default (`None`)
        no transformation is applied.

    norm_order : int, default=1
        The norm order to apply when `transform_func="norm"`. Only applied
        when `importances.ndim > 1`.

    Returns
    -------
    importances : ndarray of shape (n_features,)
        The features importances, optionally transformed.
    """
    if isinstance(getter, str):
        if getter == "auto":
            if hasattr(estimator, "coef_"):
                getter = attrgetter("coef_")
            elif hasattr(estimator, "feature_importances_"):
                getter = attrgetter("feature_importances_")
            else:
                raise ValueError(
                    "when `importance_getter=='auto'`, the underlying "
                    f"estimator {estimator.__class__.__name__} should have "
                    "`coef_` or `feature_importances_` attribute. Either "
                    "pass a fitted estimator to feature selector or call fit "
                    "before calling transform."
                )
        else:
            getter = attrgetter(getter)
    elif not callable(getter):
        raise ValueError("`importance_getter` has to be a string or `callable`")

    importances = getter(estimator)

    if transform_func is None:
        return importances
    elif transform_func == "norm":
        if importances.ndim == 1:
            importances = np.abs(importances)
        else:
            importances = np.linalg.norm(importances, axis=0, ord=norm_order)
    elif transform_func == "square":
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)
    else:
        raise ValueError(
            "Valid values for `transform_func` are "
            + "None, 'norm' and 'square'. Those two "
            + "transformation are only supported now"
        )

    return importances


def Kbest(X, y, task_type):
    if task_type == 'reg':
        score_func = f_regression
    else:
        score_func = f_classif
    skb = SelectKBest(score_func=score_func)
    score_func_ret = skb.score_func(X, y)
    if isinstance(score_func_ret, (list, tuple)):
        scores_, pvalues_ = score_func_ret
        # pvalues_ = np.asarray(pvalues_)
    else:
        scores_ = score_func_ret
        # pvalues_ = None
    scores_ = np.asarray(scores_)
    return scores_


def LASSO(X, y, task_type):
    if task_type == 'reg':
        score_func = LinearSVR(C=1.0)
    else:
        score_func = LinearSVC(C=1.0, penalty='l1', dual=False)
    model = SelectFromModel(score_func)
    model.fit(X, y)
    return get_feature_importances(model.estimator_, getter='auto')

def Rfe(X, y, task_type):
    # k = X.shape[1] / 2
    if task_type == 'reg':
        # estimator = SVR(kernel="linear")
        estimator = RandomForestRegressor(random_state=0, n_jobs=128)
    else:
        estimator = RandomForestClassifier(random_state=0, n_jobs=128)
        # estimator = RandomForestClassifier(max_depth=7, random_state=0, n_jobs=128)
    selector = RFE(estimator, n_features_to_select=0.5, step=1)
    selector.fit(X, y)
    choice = selector.get_support(True)
    imp = get_feature_importances(selector.estimator_, getter='auto')
    score = torch.zeros(X.shape[1])
    for ind, i in enumerate(choice):
        if i == 0:
            continue
        else :
            score[i] = imp[ind]
    return score


    # for i in imp:


funcs = [LASSO, Kbest, Rfe]

def rest(X, y, task_type):
    if task_type == 'reg':
        return [i(X, y, task_type) for i in funcs]
    imps = []
    dep = 12
    for method in ['RF', 'XGB', 'SVM', 'KNN', 'Ridge', 'DT']:
        if method == 'RF':
            if task_type == 'cls':
                model = RandomForestClassifier(random_state=0, n_jobs=128)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
            else:
                model = RandomForestRegressor(max_depth=dep, random_state=0, n_jobs=128)
        elif method == 'XGB':
            if task_type == 'cls':
                model = XGBClassifier(eval_metric='logloss', n_jobs=128)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=2, n_jobs=128)
                # continue
                model = XGBRegressor(eval_metric='logloss', n_jobs=128)
        elif method == 'SVM':
            if task_type == 'cls':
                model = LinearSVC()
            elif task_type == 'mcls':
                model = LinearSVC()
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=3, n_jobs=128)
                # continue
                model = LinearSVR()
        elif method == 'Ridge':
            if task_type == 'cls':
                model = RidgeClassifier()
            elif task_type == 'mcls':
                model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=5, n_jobs=128)
                # continue
                model = Ridge()
        elif method == 'LASSO':
            if task_type == 'cls':
                model = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=128)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(LogisticRegression(penalty='l1', solver='liblinear'), n_jobs=128)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=8, n_jobs=128)
                model = DecisionTreeRegressor(max_depth=7, random_state=1)
                # continue
                # model = Lasso()
        else:  # dt
            if task_type == 'cls':
                model = DecisionTreeClassifier()
            elif task_type == 'mcls':
                model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=12, n_jobs=128)
                # continue
                model = DecisionTreeRegressor(max_depth=dep)
        selector = SelectFromModel(model)
        selector.fit(X, y)
        if task_type == 'mcls':
            if method!= 'SVM':
                overall_imp = []
                for i in selector.estimator_.estimators_:
                    overall_imp.append(get_feature_importances(i, getter='auto'))
                imps.append(np.concatenate([i.reshape(-1,1) for i in overall_imp], 1).mean(1))
            else:
                overall_imp = get_feature_importances(selector.estimator_, getter='auto')
                imps.append(overall_imp.mean(0))
        else:
            score = get_feature_importances(selector.estimator_, getter='auto')
            # if task_type == 'reg':
            #     score_ = torch.LongTensor(score.argsort())
            #     choice_index = score_[:k]
            #     choice = torch.zeros(score_.shape[0])
            #     choice[choice_index] = 1
            #     test_result = fe.report_performance(choice, flag='test', store=False, rp=False)
            #     print(f'{method}', choice_index, test_result)
            imps.append(score)
    return imps


def gen_mcdm(fe: FeatureEvaluator, k):
    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]
    accumulated = rest(x, y, fe.task_type)
    norm_importance = []
    for labels in accumulated:
        labels = labels.reshape(-1)
        min_val = min(labels)
        max_val = max(labels)
        train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
        norm_importance.append(train_encoder_target)
    importances = torch.FloatTensor(norm_importance).reshape(len(norm_importance[0]), len(norm_importance))
    order = importances.argsort(descending=True)
    score = torch.zeros_like(order, dtype=torch.float)
    for index, i in enumerate(order):
        for j, pos in zip(range(order.shape[1]), i):
            score[index, pos] = (order.shape[1] - j - 1 + 0.) / order.shape[1]
    alt_name = [str(i) for i in range(x.shape[1])]
    # print(importances)
    if fe.task_type == 'reg':
        rank = mcdm.rank(importances, s_method="TOPSIS", n_method="Linear1",
                         c_method="AbsPearson",
                         w_method="VIC", alt_names=alt_name)
    else:
        rank = mcdm.rank(importances, s_method="TOPSIS", n_method="Linear1",
                     c_method="AbsPearson",
                     w_method="VIC", alt_names=alt_name)
    print('aggre', [int(i) for i, j in rank])
    selected = rank[:k]
    choice_index = torch.LongTensor([int(i) for i, score in selected])
    # info(f'current selection is {choice}')
    choice = torch.zeros(fe.ds_size)
    choice[choice_index] = 1
    test_result = fe.report_performance(choice, flag='test', store=False, rp=False)
    result = fe.report_performance(choice, flag='train', rp=False)
    info("The optimal on {} accuracy is: {}".format(fe.task_name, test_result))
    info("The optimal accuracy is: {}, the optimal selection for MCDM is:{}".format(test_result, choice))
    return result, choice


if __name__ == '__main__':
    for task_name in [
        # 'spectf', 'svmguide3', 'german_credit', 'uci_credit_card', 'spam_base', 'megawatt1','ionosphere', 'activity',
        #               'mice_protein', 'coil-20', 'minist', 'minist_fashion',
                      'openml_586', 'openml_589', 'openml_607', 'openml_616','openml_618','openml_620','openml_637']:
        load_file = f'{base_path}/history/{task_name}/fe.pkl'
        with open(load_file, 'rb') as f:
            fe = pickle.load(f)
        k = -1
        p = -1
        o = None
        for r in fe.records.r_list:
            k_ = r.operation.sum()
            p_ = r.performance
            if p_ > p:
                p = p_
                o = r.operation
                k = int(k_)
        info(str(k))
        k = int(fe.ds_size /2)
        info(f'the k for task : {task_name} is {k}')
        res, optimal_set = gen_mcdm(fe, k=k)
        # best_train = fe.generate_data(optimal_set, flag='train')
        # best_test = fe.generate_data(optimal_set, flag='test')
        # best_train.to_hdf(f'{base_path}/history/mcdm_train.hdf', key=f'{task_name}')
        # best_test.to_hdf(f'{base_path}/history/mcdm_test.hdf', key=f'{task_name}')
        # fe.report_performance(o, flag='test')
        # print(o)
        # fe.report_performance(optimal_set, flag='test')
        # print(optimal_set)
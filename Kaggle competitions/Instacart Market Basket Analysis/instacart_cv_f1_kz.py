import gc
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import os
import json
import sklearn.metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.sparse import dok_matrix, coo_matrix
from sklearn.utils.multiclass import  type_of_target

import itertools
from tqdm import tqdm


class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)




def create_products(df):
    products = df.product_id.values
    products_sort = products[df.prediction == 1]
    best_prediction = []

    if len(products_sort)==0: 
    	best_prediction = ['None'] 
    else:	
    	best_prediction += [str(p) for p in products_sort]

    best = ' '.join(best_prediction)
    return (df.iloc[0,0], best)

def create_products_faron(df):
    products = df.product_id.values
    prob = df.prediction.values

    sort_index = np.argsort(prob)[::-1]
    prob = prob[sort_index]
    products = products[sort_index]

    opt = F1Optimizer.maximize_expectation(prob)
    
    best_prediction = ['None'] if opt[1] else []
    best_prediction += [str(p) for p in products[:opt[0]]]
    f1_max = opt[2]

    best = ' '.join(best_prediction)
    return (df.iloc[0,0], best)



def multilabel_fscore(y_true, y_pred):
    """
    ex1:
    y_true = [1, 2, 3]
    y_pred = [2, 3]
    return: 0.8
    
    ex2:
    y_true = ["None"]
    y_pred = [2, "None"]
    return: 0.666
    
    ex3:
    y_true = [4, 5, 6, 7]
    y_pred = [2, 4, 8, 9]
    return: 0.25
    
    """
    y_true, y_pred = set(y_true), set(y_pred)
    
    precision = sum([1 for i in y_pred if i in y_true]) / len(y_pred)
    
    recall = sum([1 for i in y_true if i in y_pred]) / len(y_true)
    
    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)



# main part: 
if __name__ == '__main__':

    unique_orders = np.unique(order_train.order_id)
    orders_train, orders_test = train_test_split(unique_orders, test_size=0.25, random_state=2017)

    order_test = order_train.loc[np.in1d(order_train.order_id, orders_test)]
    order_train = order_train.loc[np.in1d(order_train.order_id, orders_train)]

    data = order_train[features]
    labels = order_train.iloc[:,2].values.astype(np.float32).flatten()
 
    data_val = order_test[features]
    labels_val = order_test.iloc[:,2].values.astype(np.float32).flatten()

# ...
#training part: 
# ...



    prediction = gbm.predict(data_val)
    orders = order_test.order_id.values
    products = order_test.product_id.values

    result = pd.DataFrame({'product_id': products, 'order_id': orders, 'prediction': prediction})
    print(result.head(2))
    result_val = pd.DataFrame({'product_id': products, 'order_id': orders, 'prediction': labels_val})
    print(result_val.head(2))

    data_pred = result #pd.read_pickle('data/prediction_lgbm_leaves256.pkl')
    data_cv_val = result_val

    data_pred = data_pred.loc[data_pred.prediction > 0.01, ['order_id', 'prediction', 'product_id']]
    print(data_pred.head(2))
    data_cv_val = data_cv_val.loc[data_cv_val.prediction > -1, ['order_id', 'prediction', 'product_id']]
    print(data_cv_val.head(2))

    out = [create_products_faron(group) for name, group in tqdm(data_pred.groupby(data_pred.order_id))]
    out_cv_val = [create_products(group) for name, group in tqdm(data_cv_val.groupby(data_cv_val.order_id))]
    #print(out)
    #print(out_cv_val)

    data_pred_final = pd.DataFrame(data=out, columns=['order_id', 'products'])
    print(data_pred_final.head(5))
    data_cv_final = pd.DataFrame(data=out_cv_val, columns=['order_id', 'products'])
    print(data_cv_final.head(5))

    #data_pred.to_csv('data/test11_faron_leaves256.csv', index=False)

    #f1_score_final = np.mean([multilabel_fscore(x, y) for x, y in zip(data_cv_final.products.split(), data_pred_final.products.split())])

    f1_score_final = 0
    for i in range(data_cv_final.shape[0]):
    	f1_score_final = f1_score_final+ multilabel_fscore(data_cv_final.loc[i,'products'].split(),data_pred_final.loc[i,'products'].split())

    f1_score_mean = f1_score_final / data_cv_final.shape[0]

    print('cv result f1 score: ')
    print(f1_score_mean)


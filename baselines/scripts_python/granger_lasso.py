import numpy as np
from sklearn.linear_model import LassoCV
import pandas as pd
# import networkx as nx


def granger_lasso(data, sig_level=0.05, maxlag=5, cv=5):
    '''
    Granger causality test for multi-dimensional time series
    Parameters:
    -----------
    data - input data (nxd)
    maxlag: maximum time lag
    cv: number of cross validation folds for lasso regression
    Returns:
    ----------
    coeff: coefficient matrix [A_1, A_2, ..], where A_k is the dxd causal matrix for the k-th time lag. The ij-th entry
    in A_k represents the causal influence from j-th variable to the i-th variable.
    '''

    n, dim = data.shape
    # stack data to form one-vs-all regression
    Y = data.values[maxlag:]
    X = np.hstack([data.values[maxlag - k:-k] for k in range(1, maxlag + 1)])

    lasso_cv = LassoCV(cv=cv)
    coeff = np.zeros((dim, dim * maxlag))
    # Consider one variable after the other as target
    for i in range(dim):
        lasso_cv.fit(X, Y[:, i])
        coeff[i] = lasso_cv.coef_

    print(coeff)
    names = data.columns
    dataset = pd.DataFrame(np.zeros((len(names), len(names)), dtype=int), columns=names, index=names)
    # g = nx.DiGraph()
    # g.add_nodes_from(names)
    for i in range(dim):
        for l in range(maxlag):
            for j in range(dim):
                # if abs(coeff[i, j+l*dim]) > sig_level:
                if abs(coeff[i, j + l * dim]) > 0:
                    # g.add_edge(names[j], names[i])
                    dataset[names[i]].loc[names[j]] = 2

    for c in dataset.columns:
        for r in dataset.index:
            if dataset[r].loc[c] == 2:
                if dataset[c].loc[r] == 0:
                    dataset[c].loc[r] = 1
            if r == c:
                dataset.loc[r, c] = 1
    return dataset


if __name__ == "__main__":
    # path = "../../../../data/simulated_ts_data/fork/data_"+str(0)+".csv"
    path = "../../data/simulated_ts_data_v2/acyclic/diamond/datasets/dataset_selfcause=None_"+str(0)+".csv"
    data = pd.read_csv(path, index_col=0, header=0, delimiter=',')
    data = data.reset_index(drop=True)
    data = data.loc[:1000]
    print(data)

    df = granger_lasso(data, 2)
    print(df)

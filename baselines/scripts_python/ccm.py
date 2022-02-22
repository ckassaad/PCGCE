import pandas as pd
from baselines.scripts_python.python_packages.CITCE.citce import PCTCE
import networkx as nx

import skccm
from skccm.utilities import train_test_split
import numpy as np


def ccm(data, sig_level=0.05, nlags=5, verbose=True):
    embed = 2
    for col1 in data.columns:
        e1 = skccm.Embed(data[col1])
        x1 = e1.embed_vectors_1d(nlags, embed)
        for col2 in data.columns:
            e2 = skccm.Embed(data[col2])
            x2 = e2.embed_vectors_1d(nlags, embed)
            x1tr, x1te, x2tr, x2te = skccm.utilities.train_test_split(x1, x2, percent=.75)

            cm = skccm.CCM()
            len_tr = len(x1tr)
            lib_lens = np.arange(10, len_tr, len_tr / 20, dtype='int')

            # test causation
            cm.fit(x1tr, x2tr)
            x1p, x2p = cm.predict(x1te, x2te, lib_lengths=lib_lens)

            sc1, sc2 = cm.score()

            sc_diff = sc2[0] - sc1[0]
            print(col1, col2)
            print(sc_diff)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return 1


if __name__ == "__main__":
    import os
    structure = "diamond"
    print(os.getcwd())
    path = "../../data/simulated_ts_data/"+str(structure)+"/data_"+str(0)+".csv"
    data = pd.read_csv(path, delimiter=',', index_col=0)
    data = data.loc[:200]
    print(data)
    graphs = ccm(data, sig_level=0.05, nlags=5, verbose=True)

from baselines.scripts_python.python_packages.tigramiteNew.tigramite.pcmci import PCMCI
from baselines.scripts_python.python_packages.tigramiteNew.tigramite.independence_tests import ParCorr, CMIknn
from baselines.scripts_python.python_packages.tigramiteNew.tigramite import data_processing as pp
import numpy as np
import pandas as pd


def pcmciplus(data, tau_max=5, cond_ind_test="CMIknn", alpha=0.05):
    if cond_ind_test == "CMIknn":
        cond_ind_test = CMIknn()
    elif cond_ind_test == "ParCorr":
        cond_ind_test = ParCorr()

    data_tigramite = pp.DataFrame(data.values, var_names=data.columns)

    pcmci = PCMCI(
        dataframe=data_tigramite,
        cond_ind_test=cond_ind_test,
        verbosity=1)
    # res = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha)
    res = pcmci.run_pcmciplus(selected_links=None,
                      tau_min=0,
                      tau_max=tau_max,
                      pc_alpha=alpha,
                      contemp_collider_rule='majority',
                      conflict_resolution=True,
                      reset_lagged_links=False,
                      max_conds_dim=None,
                      max_conds_py=None,
                      max_conds_px=None,
                      max_conds_px_lagged=None,
                      fdr_method='none',
                      )
    res_dict = dict()
    graph = res['graph']
    sig_links = (graph != "") * (graph != "<--")
    for j in range(data.shape[1]):
        res_dict[pcmci.var_names[j]] = []
        links = {(p[0], -p[1]): np.abs(res['val_matrix'][p[0], j, abs(p[1])]) for p in zip(*np.where(sig_links[:, j, :]))}
        sorted_links = sorted(links, key=links.get, reverse=True)
        for p in sorted_links:
            print(pcmci.var_names[p[0]], pcmci.var_names[j], p[1])
            res_dict[pcmci.var_names[j]].append((pcmci.var_names[p[0]], p[1]))
    print(res_dict)

    # res_summary_array = np.zeros([data.shape[1], data.shape[1]])
    #
    # for k in pcmci.all_parents.keys():
    #     temp = pcmci.all_parents[k]
    #     temp = np.unique([x[0] for x in temp])
    #     for i in temp:
    #         if k == i:
    #             res_summary_array[k, i] = 1
    #         else:
    #             if res_summary_array[k, i] == 0:
    #                 res_summary_array[k, i] = 1
    #             res_summary_array[i, k] = 2
    # res_summary_array = pd.DataFrame(res_summary_array, columns=data.columns, index=data.columns)
    return res_dict

def v_structure(N):
    N = N + 1
    print("v_structure: 0 -> 2 <- 1")
    epsw = np.random.randn(N) ** 1
    epsx = np.random.randn(N) ** 1
    epsy = np.random.randn(N) ** 1

    x = epsx
    y = epsy
    w = 0.9 * x + 0.8 * y + 0.2 * epsw

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    data = data.drop(data.index[0])
    data = data.reset_index(drop=True)
    return data

if __name__ == "__main__":
    import pandas as pd
    structure = "v_structure_0"
    path = "../../data/simulated_ts_data_v2/acyclic/"+str(structure)+"/datasets/dataset_selfcause=0_2.csv"
    # data = pd.read_csv(path, delimiter=',', index_col=0)
    data = v_structure(1000)
    model = pcmciplus(data)
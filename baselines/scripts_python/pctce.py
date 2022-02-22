import pandas as pd
from baselines.scripts_python.python_packages.CITCE.citce import PCTCE
import networkx as nx


def pctce(data, sig_level=0.05, nlags=5, verbose=True):
    citce = PCTCE(data, sig_lev=sig_level, lag_max=nlags, p_value=True, rank_using_p_value=False, verbose=verbose,
               num_processor=4)
    _ = citce.fit()
    g = citce.graph.to_summary()

    nodes = list(citce.graph.map_names_nodes.keys())
    # map_inverse = dict()
    # for node in nodes:
    #     for t_node in citce.graph.map_names_nodes[node]:
    #         map_inverse[t_node] = node

    # g = nx.DiGraph()
    # g.add_nodes_from(nodes)
    og = nx.DiGraph()
    sg = nx.DiGraph()
    og.add_nodes_from(nodes)
    sg.add_nodes_from(nodes)
    for cause, effects in g.adj.items():
        for effect, _ in effects.items():
            if cause != effect:
                og.add_edges_from([(cause, effect)])
            else:
                sg.add_edges_from([(cause, effect)])

    # for cause, effects in citce.graph.ghat.adj.items():
    #     for effect, _ in effects.items():
    #         if map_inverse[cause] != map_inverse[effect]:
    #             og.add_edges_from([(map_inverse[cause], map_inverse[effect])])
    #             g.add_edges_from([(map_inverse[cause], map_inverse[effect])])
    #         else:
    #             sg.add_edges_from([(map_inverse[cause], map_inverse[effect])])
    #             g.add_edges_from([(map_inverse[cause], map_inverse[effect])])
    print(g.edges)
    print(og.edges)
    print(sg.edges)
    return g, og, sg


if __name__ == "__main__":
    import os
    structure = "diamond"
    print(os.getcwd())
    path = "../../data/simulated_ts_data/"+str(structure)+"/data_"+str(0)+".csv"
    data = pd.read_csv(path, delimiter=',', index_col=0)
    data = data.loc[:200]
    print(data)
    graphs = pctce(data, sig_level=0.05, nlags=5, verbose=True)
    print(graphs[0].edges)
    print(graphs[1].edges)
    print(graphs[2].edges)

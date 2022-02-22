#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Discovery of summary causal graphs for time series: script implementing
the PCTMI and FCITMI methods.
Parallelization is done across variables for the skeleton construction step and for the rule origin of causality.

Date: Dec 2019
Author: Karim Assaad, karimassaad3@gmail.com, karim.assaad@univ.grenoble.alpes.fr, karim.assaad@coservit.com
paper: soon
"""

import networkx as nx
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from datetime import datetime

from baselines.scripts_python.python_packages.CITCE.tigramite.tigramite.independence_tests import CMIknn
from baselines.scripts_python.python_packages.CITCE.tigramite.tigramite.independence_tests import ParCorr

from tqdm import tqdm


class PseudoSummaryGraph:
    """
    Graph structure
    0: no edge
    1: a tail -
    2: arrow head ->
    """
    def __init__(self, nodes):
        super(PseudoSummaryGraph, self).__init__()
        self.nodes_present, self.nodes_past, self.map_names_nodes = self.get_nodes(nodes)
        self.ghat = nx.DiGraph()
        self.ghat.add_nodes_from(self.nodes_present + self.nodes_past)

        for node_present in self.nodes_present:
            for node_past in self.nodes_past:
                self.ghat.add_edges_from([(node_past, node_present)])
            for node_present_2 in self.nodes_present:
                if node_present != node_present_2:
                    self.ghat.add_edges_from([(node_present_2, node_present)])

        self.d = len(nodes)*2
        self.sep = dict()
        for tup in self.ghat.edges:
            self.sep[tup] = []

    @staticmethod
    def get_nodes(names):
        nodes_present = []
        nodes_past = []
        map_names_nodes = dict()
        for name_p in names:
            try:
                int(name_p)
                node_p_present = "V" + str(name_p) + '_t'
                node_p_past = "V" + str(name_p) + '_t-'
                nodes_present.append(node_p_present)
                nodes_past.append(node_p_past)
                map_names_nodes[name_p] = [node_p_past, node_p_present]

            except ValueError:
                node_p_present = str(name_p) + '_t'
                node_p_past = str(name_p) + '_t-'
                nodes_present.append(node_p_present)
                nodes_past.append(node_p_past)
                map_names_nodes[name_p] = [node_p_past, node_p_present]
        return nodes_present, nodes_past, map_names_nodes

    def add_sep(self, node_p, node_q, node_r):
        """
        :param p: index of a time series
        :param q: index of a time series
        :param r: index of seperation set
        """
        if node_p in self.ghat.predecessors(node_q):
            self.sep[(node_p, node_q)].append(node_r)
        if node_q in self.ghat.predecessors(node_p):
            self.sep[(node_q, node_p)].append(node_r)

    def number_par_all(self):
        """
        :return: dict of number of adjacencies per node
        """
        dict_num_adj = dict()
        for node_p in self.ghat.nodes:
            dict_num_adj[node_p] = len(list(self.ghat.predecessors(node_p)))
        return dict_num_adj

    def to_summary(self):
        """
        :return: summary graph
        """
        map_names_nodes_inv = dict()
        nodes = list(self.map_names_nodes.keys())
        for node in nodes:
            for node_t in self.map_names_nodes[node]:
                map_names_nodes_inv[node_t] = node

        ghat_summary = nx.DiGraph()
        ghat_summary.add_nodes_from(nodes)

        # unique_edges = []
        for (node_p_t, node_q_t) in self.ghat.edges:
            node_p, node_q = map_names_nodes_inv[node_p_t], map_names_nodes_inv[node_q_t]
            # if (node_p, node_q) not in unique_edges:
            if (node_p, node_q) not in ghat_summary.edges:
                # unique_edges.append((node_p, node_q))
                ghat_summary.add_edges_from([(node_p, node_q)])
        print(ghat_summary.edges)
        return ghat_summary

class RankingList:
    def __init__(self):
        self.val = np.array([])
        self.elem_p = np.array([])
        self.elem_q = np.array([])
        self.elem_r = []

    def add(self, node_p, node_q, val, nodes_r):
        """
        :param p: index of a time series
        :param q: index of a time series
        :param val: value of mutual information
        :param r: index of set of conditionals
        """
        self.val = np.append(self.val, val)
        self.elem_p = np.append(self.elem_p, node_p)
        self.elem_q = np.append(self.elem_q, node_q)
        self.elem_r.append(nodes_r)

    def sort(self, descending=True):
        """
        :param descending: (bool) sort ascending vs. descending. By default True
        """
        idx = np.argsort(self.val)
        if descending:
            idx = np.flip(idx)
        self.val = np.take_along_axis(self.val, idx, axis=0)
        self.elem_p = np.take_along_axis(self.elem_p, idx, axis=0)
        self.elem_q = np.take_along_axis(self.elem_q, idx, axis=0)
        sorted_elem_r = []
        for i in idx:
            sorted_elem_r.append(self.elem_r[i])
        self.elem_r = sorted_elem_r


def gtce(x, y, z=None, sampling_rate_tuple=(1, 1, 1), p_value=True, k=10):
    '''
    Greedy temporal causation entropy
    :param x:
    :param y:
    :param z:
    :param sampling_rate_tuple:
    :param p_value:
    :return:
    '''
    measure = "cmiknn"
    if measure == "cmiknn":
        cd = CMIknn(mask_type=None, significance='shuffle_test', fixed_thres=None, sig_samples=10000,
                    sig_blocklength=3, knn=k, confidence='bootstrap', conf_lev=0.9, conf_samples=10000,
                    conf_blocklength=1, verbosity=0)
    # elif measure == "parcorr":
    #     cd = ParCorr(mask_type=None, significance='shuffle_test', fixed_thres=None, sig_samples=10000,
    #                  sig_blocklength=3, confidence='bootstrap', conf_lev=0.9, conf_samples=10000, conf_blocklength=1,
    #                  verbosity=0)
    else:
        cd = None
        print("Independence measure '" + str(measure) + "' do not exist.")
        exit(0)
    dim_x = x.shape[1]
    dim_y = y.shape[1]
    if z is not None:
        z_df = pd.DataFrame()
        for k in z.keys():
            if isinstance(z[k], pd.Series):
                z[k] = z[k].to_frame()
            z_df[z[k].columns] = z[k].reset_index(drop=True)

        dim_z = z_df.shape[1]
        X = np.concatenate((x.values, y.values, z_df.values), axis=1)
        xyz = np.array([0] * dim_x + [1] * dim_y + [2] * dim_z)
    else:
        X = np.concatenate((x, y), axis=1)
        xyz = np.array([0] * dim_x + [1] * dim_y)

    value = cd.get_dependence_measure(X.T, xyz)
    if p_value:
        pvalue = cd.get_shuffle_significance(X.T, xyz, value)
    else:
        pvalue = value

    return pvalue, value


def get_sampling_rate(ts):
    # index of all non nan values in time series
    idx = np.argwhere(~np.isnan(ts).values)
    if len(idx) == len(ts):
        return True, 1
    # differentiate index, if there's no nan all values should be equal to 1
    diff = np.diff(idx, axis=0)
    udiff = np.unique(diff)
    if (len(udiff) == 1) and (udiff != 1):
        cd_bool = True
        cd_value = int(udiff)
    elif len(udiff) == 2:
        idx = np.argwhere(diff.reshape(-1) > 1)
        diff = diff[idx]
        udiff = np.unique(diff)
        if len(udiff) == 1:
            cd_bool = True
            cd_value = int(udiff)
        else:
            # ???
            cd_bool = False
            cd_value = np.nan
    else:
        # if there is no delay in the time series
        cd_bool = False
        cd_value = np.nan
    return cd_bool, cd_value


def get_alpha(mts, k=10):
    mi_list = []
    for i in range(mts.shape[1]):
        for t in range(100):
            ts_i = mts[mts.columns[i]].dropna().to_frame()
            ts_j = 0.05*ts_i + 0.95*np.random.randn(ts_i.shape[0], ts_i.shape[1])
            pval, val = gtce(ts_i, ts_j, k=k, p_value=False)
            mi_list.append(val)
    alpha = abs(max(mi_list))
    return alpha


class CITCE:
    def __init__(self, series, sig_lev=0.05, lag_max=5, p_value=True, rank_using_p_value=False, verbose=True, num_processor=-1):
        """
        Causal inference (Wrapper) using TCE (contain functions for skeleton construction)
        :param series: d-time series (with possibility of different sampling rate)
        :param sig_lev: significance level. By default 0.05
        :param p_value: Use p_value for decision making. By default True
        :param verbose: Print results. By default: True
        :param num_processor: number of processors for parallelization. By default -1 (all)
        """
        self.series = series
        self.n = series.shape[0]
        self.d = series.shape[1]
        self.names = self.series.columns
        self.graph = PseudoSummaryGraph(self.names)
        self.num_processor = num_processor
        self.p_value = p_value
        self.lag_max = lag_max
        if self.p_value == rank_using_p_value:
            self.rank_using_p_value = rank_using_p_value
        elif not rank_using_p_value:
            self.rank_using_p_value = rank_using_p_value
        else:
            print("Warning: rank_using_p_value can be True iff p_value is True. Using rank_using_p_value=False")
            self.rank_using_p_value = False
        self.verbose = verbose

        self.data_dict = dict()

        self.sampling_rate = dict()
        for name_p in self.names:
            _, s_r = get_sampling_rate(self.series[name_p])
            name_p_past, name_p_present = self.graph.map_names_nodes[name_p]
            self.sampling_rate[name_p_past] = s_r
            self.sampling_rate[name_p_present] = s_r

        self.sig_lev = sig_lev
        self.alpha = get_alpha(series)

        for name_p in self.names:
            node_p_past, node_p_present = self.graph.map_names_nodes[name_p]
            self.data_dict[node_p_past], self.data_dict[node_p_present] = self.window_representation_past_present(name_p, node_p_past, node_p_present)

        self.mi_df = pd.DataFrame(np.ones([self.graph.d, self.graph.d]), columns=self.graph.ghat.nodes, index=self.graph.ghat.nodes)
        self.cmi_df = pd.DataFrame(np.ones([self.graph.d, self.graph.d]), columns=self.graph.ghat.nodes, index=self.graph.ghat.nodes)

        if self.verbose:
            print("n: "+str(self.n))
            print("d: "+str(self.d))
            print("names: "+str(self.names))
            print("sampling_rate: "+str(self.sampling_rate))
            print("significance level:"+str(self.sig_lev))
            print("alpha:"+str(self.alpha))

    def window_representation_past_present(self, name_p, node_p_past, node_p_present, overlap=True):
        ts = self.series[name_p].dropna()
        ts_window_past = pd.DataFrame()
        if self.lag_max == 0:
            ts_present = ts.rename(node_p_present).to_frame()
        else:
            for i in range(self.lag_max):
                i_data = ts[i:(ts.shape[0] - self.lag_max + i)].values
                ts_window_past.loc[:, node_p_past + str(self.lag_max - i)] = i_data
            ts_present = ts.rename(node_p_present).to_frame()
            ts_present = ts_present.loc[self.lag_max:]
            if not overlap:
                ts_window_past = ts_window_past.iloc[::ts_window_past, :]
        return ts_window_past, ts_present.reset_index(drop=True)

    def _mi_pq(self, node_p, node_q):
        """
        estimate tmi between two time series
        :param p: time series with index p
        :param q: time series with index q
        :return: p, q and the estimated value of tmi(p,q)
        """
        x = self.data_dict[node_p]
        y = self.data_dict[node_q]

        mi_pval, mi_val = gtce(x, y, z=None, sampling_rate_tuple=(self.sampling_rate[node_p], self.sampling_rate[node_q]), p_value=self.p_value)
        return node_p, node_q, mi_pval

    def skeleton_initialize(self):
        """
        initialize graph, remove all unconditional independencies and rank neighbors
        """
        if self.verbose:
            print("######################################")
            print("Skeletion Initialization")
            print("######################################")

        unique_edges = []
        for (node_p, node_q) in self.graph.ghat.edges:
            if (node_q, node_p) not in unique_edges:
                unique_edges.append((node_p, node_q))

        res = Parallel(n_jobs=self.num_processor)(delayed(self._mi_pq)(node_p, node_q) for node_p, node_q in tqdm(unique_edges))
        # res = []
        # for (node_p, node_q) in tqdm(unique_edges):
        #     res.append(self._mi_pq(node_p, node_q))

        for pq in range(len(res)):
            node_p, node_q, mi = res[pq][0], res[pq][1], res[pq][2]
            self.mi_df[node_p].loc[node_q] = mi
            self.mi_df[node_q].loc[node_p] = mi
            if self.verbose:
                print("p=" + node_p + "; q=" + node_q + "; I(p,q)=" + "{: 0.5f}".format(self.mi_df[node_p].loc[node_q]), end=" ")
            if self.p_value:
                if (self.data_dict[node_p].shape[1] > 1) or (self.data_dict[node_q].shape[1] > 1):
                    test = self.mi_df[node_p].loc[node_q] > self.sig_lev * 2
                else:
                    test = self.mi_df[node_p].loc[node_q] > self.sig_lev
            else:
                test = self.mi_df[node_p].loc[node_q] < self.alpha
            if test:
                if self.verbose:
                    print("=> Remove link between "+node_p+" and "+node_q)
                self.graph.ghat.remove_edge(node_p, node_q)
                if (node_p in self.graph.nodes_present) and (node_q in self.graph.nodes_present):
                    self.graph.ghat.remove_edge(node_q, node_p)
            else:
                if self.verbose:
                    print()

    def _cmi_sep_set_pq(self, node_p, node_q, set_size):
        """
        estimate ctmi between two time series conditioned on each set of neighbors with cardinality equal to set_size
        :param p: time series with index p
        :param q: time series with index q
        :param set_size: cardinality of the set of neighbors
        :return: p, q, list if estimated value of gtce(p,q,r_set), and list of all r_sets
        """
        v_list = []
        nodes_r = list(set(list(self.graph.ghat.predecessors(node_p)) + list(self.graph.ghat.predecessors(node_q))))
        if node_p in nodes_r:
            nodes_r.remove(node_p)
        if node_q in nodes_r:
            nodes_r.remove(node_q)
        nodes_r = [list(r) for r in itertools.combinations(nodes_r, set_size)]

        x = self.data_dict[node_p]
        y = self.data_dict[node_q]

        for set_nodes_r in nodes_r:
            z = dict()
            for node_r in set_nodes_r:
                z[node_r] = self.data_dict[node_r]
            cmi_pval, cmi_val = gtce(x, y, z, self.sampling_rate, p_value=self.rank_using_p_value)

            if self.rank_using_p_value:
                v_list.append(cmi_pval)
            else:
                v_list.append(cmi_val)
        if v_list:
            return node_p, node_q, v_list, nodes_r

    def rank_cmi_sep_set_parallel(self, set_size):
        """
        rank pairs of time series based on the estimation of gtce between each pair of connected nodes
        :param set_size: cardinality of the set of neighbors
        :return: ranking of each pair of connected nodes based gtce
        """
        print("set_size " + str(set_size))

        dict_num_adj = self.graph.number_par_all()
        unique_qualified_edges = []
        for (node_p, node_q) in self.graph.ghat.edges:
            if (node_q, node_p) not in unique_qualified_edges:
                if dict_num_adj[node_p] + dict_num_adj[node_q] > set_size:
                    unique_qualified_edges.append((node_p, node_q))

        res = Parallel(n_jobs=self.num_processor)(delayed(self._cmi_sep_set_pq)(node_p, node_q, set_size) for node_p, node_q in
                                                  tqdm(unique_qualified_edges))
        # res = []
        # for (node_p, node_q) in tqdm(unique_qualified_edges):
        #     res.append(self._cmi_sep_set_pq(node_p, node_q, set_size))

        ranks = RankingList()
        for pq in range(len(res)):
            if res[pq] is not None:
                if isinstance(res[pq][2], list):
                    for r in range(len(res[pq][2])):
                        ranks.add(res[pq][0], res[pq][1], res[pq][2][r], res[pq][3][r])
                else:
                    ranks.add(res[pq][0], res[pq][1], res[pq][2], res[pq][3])
        if self.rank_using_p_value:
            ranks.sort(descending=True)
        else:
            ranks.sort(descending=False)
        return ranks

    def find_sep_set(self):
        """
        find the most contributing separation set (if it exists) between each pair of nodes
        """
        if self.verbose:
            print("######################################")
            print("Skeletion Speperation")
            print("######################################")

        print("max set size = " + str(self.graph.d-1))
        for set_size in range(1, self.graph.d-1):
            ranks = self.rank_cmi_sep_set_parallel(set_size)
            if self.verbose:
                print("Ranking:")
                print("p: "+str(ranks.elem_p))
                print("q: " + str(ranks.elem_q))
                print("r: " + str(ranks.elem_r))
                print("val: " + str(ranks.val))
            for node_p, node_q, r_set, cmi in zip(ranks.elem_p, ranks.elem_q, ranks.elem_r, ranks.val):
                test = (node_p in self.graph.ghat.predecessors(node_q)) or (node_q in self.graph.ghat.predecessors(node_p))
                for node_r in r_set:
                    if not test:
                        break
                    test = test and ((node_r in self.graph.ghat.predecessors(node_q)) or (node_r in self.graph.ghat.predecessors(node_p)))
                if test:
                    mi = self.mi_df[node_p].loc[node_q]

                    if self.p_value != self.rank_using_p_value:
                        x = self.data_dict[node_p]
                        y = self.data_dict[node_q]

                        z = dict()
                        multi_dim = False
                        for node_r in r_set:
                            z[node_r] = self.data_dict[node_r]
                            if node_r in self.graph.nodes_past:
                                multi_dim = True
                        cmi, _ = gtce(x, y, z, self.sampling_rate, p_value=self.p_value)
                    if self.verbose:
                        print("p=" + node_p + "; q=" + node_q + "; r=" + str(r_set) + "; I(p,q|r)=" + "{: 0.5f}".format(
                            cmi) + "; I(p,q)=" + "{: 0.5f}".format(mi), end=" ")

                    if self.p_value:
                        if (self.data_dict[node_p].shape[1] > 1) or (self.data_dict[node_q].shape[1] > 1) or multi_dim:
                            test = mi < self.sig_lev * 2 < cmi
                        else:
                            test = mi < self.sig_lev < cmi
                    else:
                        test = cmi < self.alpha
                    if test:
                        self.cmi_df[node_p].loc[node_q] = cmi
                        self.cmi_df[node_q].loc[node_p] = cmi
                        if self.verbose:
                            print("=> remove link between " + node_p + " and " + node_q)

                        for node_r in r_set:
                            self.graph.add_sep(node_p, node_q, node_r)

                        self.graph.ghat.remove_edge(node_p, node_q)
                        if (node_p in self.graph.nodes_present) and (node_q in self.graph.nodes_present):
                            self.graph.ghat.remove_edge(node_q, node_p)

                    else:
                        if self.verbose:
                            print()


class PCTCE(CITCE):
    def __init__(self, series, sig_lev=0.05, lag_max=5, p_value=True, rank_using_p_value=False, verbose=True, num_processor=-1):
        """
        PC for time series using GTCE
        :param series: d-time series (with possibility of different sampling rate)
        :param sig_lev: significance level. By default 0.05
        :param p_value: Use p_value for decision making. By default True
        :param verbose: Print results. By default: True
        :param num_processor: number of processors for parallelization. By default -1 (all)
        """
        CITCE.__init__(self, series, sig_lev, lag_max, p_value, rank_using_p_value, verbose, num_processor)

    def _find_shortest_directed_paths_util(self, p, q, visited, path, all_path):
        """
        sub function of _find_shortest_directed_paths
        :param p: index of time series
        :param q: index of time series
        :param visited: list of visited nodes
        :param path: current path
        :param all_path: list of all discovered paths
        """
        # Mark the current node as visited and store in path
        visited[p] = True
        path.append(p)

        # If current vertex is same as destination, then print
        # current path[]
        if p == q:
            if len(path) > 2:
                all_path.append(path.copy()[1:-1])
                return path
        else:
            # If current vertex is not destination
            # Recur for all the vertices child of this vertex
            child_p = np.where(self.graph.edges[p, :] == 2)[0]
            for k in child_p:
                if not visited[k]:
                    self._find_shortest_directed_paths_util(k, q, visited, path, all_path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[p] = False

    def _find_shortest_directed_paths(self, p, q):
        """
        find shortest directed path between time series of index p and time series of index q
        :param p: index of time series
        :param q: index of time series
        :return: all directed paths from p to q
        """
        # Mark all the vertices as not visited
        visited = [False] * self.d

        # Create an array to store paths
        path = []
        all_path = []

        # Call the recursive helper function to print all paths
        self._find_shortest_directed_paths_util(p, q, visited, path, all_path)
        return all_path

    def rule_origin_causality(self):
        """
        rule 0 (origin of causality) from PC
        """
        if self.verbose:
            print("######################################")
            print("Rule Origin of Causality")
            print("######################################")

        edges_t = [(node_p, node_r) for (node_p, node_r) in self.graph.ghat.edges if (
                (node_p, node_r) in self.graph.ghat.edges) and (node_p in self.graph.nodes_present) and (
                node_r in self.graph.nodes_present)]
        for (node_p_t, node_r_t) in edges_t:
            nodes_q = [node_q for node_q in self.graph.ghat.predecessors(node_r_t) if (node_q != node_p_t) and (
                    node_q not in self.graph.ghat.predecessors(node_p_t)) and (
                    node_q not in self.graph.ghat.successors(node_p_t)) and (
                    node_r_t not in self.graph.sep[(node_q, node_p_t)])]
            for node_q in nodes_q:
                print(node_p_t, node_r_t, node_q)
                if ((node_q, node_r_t) in self.graph.ghat.edges) and ((node_p_t, node_r_t) in self.graph.ghat.edges):
                    if (node_r_t, node_p_t) in self.graph.ghat.edges:
                        self.graph.ghat.remove_edge(node_r_t, node_p_t)
                    if node_q in self.graph.nodes_present:
                        if (node_r_t, node_q) in self.graph.ghat.edges:
                            self.graph.ghat.remove_edge(node_r_t, node_q)

    def rule_2(self):
        """
        rule 2 from PC
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 3")
            print("######################################")
        test_find_orientation = False

        for i in range(self.graph.d):
            j_list = np.where(self.graph.edges[i, :] == 1)[0].tolist()
            if i in j_list:
                j_list.remove(i)
            for j in j_list:
                if self.graph.edges[j, i] == 1:
                    shortest_directed_path = self._find_shortest_directed_paths(i, j)
                    if len(shortest_directed_path) > 0:
                        self.graph.edges[i, j] = 2
                        test_find_orientation = True
                        if self.verbose:
                            print_path = '->'.join(map(str, shortest_directed_path[0]))
                            print(str(i)+"-"+str(j)+" and "+str(i) + "->" + print_path + "->" + str(j), end=" ")
                            print("=> orient " + str(i) + "->" + str(j))
        return test_find_orientation

    def rule_3(self):
        """
        rule 3 from PC
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 4")
            print("######################################")

        test_find_orientation = False

        for i in range(self.graph.d):
            for j in range(i + 1, self.graph.d):
                if (self.graph.edges[i, j] == 0) and (self.graph.edges[j, i] == 0):
                    colliders = [k for k in range(self.graph.d) if (k != i) and (k != j) and (
                            (self.graph.edges[j, k] == 2) and (self.graph.edges[i, k] == 2))]
                    k_list = [k for k in range(self.graph.d) if (k != i) and (k != j) and (
                            (self.graph.edges[j, k] == 1) and (self.graph.edges[i, k] == 1))
                              and (self.graph.edges[k, j] == 1) and (self.graph.edges[k, i] == 1)]
                    if len(colliders) > 0 and len(k_list) > 0:
                        for c in colliders:
                            for k in k_list:
                                if (self.graph.edges[c, k] == 1) and (self.graph.edges[k, c] == 1):
                                    test_find_orientation = True
                                    self.graph.edges[k, c] = 2
                                    if self.verbose:
                                        print(str(i) + "->" + str(c) + "<-" + str(j) + " and " + str(i) + "-" +
                                              str(k) + "-" + str(j) + " and " + str(k) + "-" + str(c),
                                              end=" ")
                                        print("=> orient " + str(k) + "->" + str(c))
        return test_find_orientation

    def fit(self):
        """
        run PCTCE
        :return: graph (CPDAG)
        """
        if self.verbose:
            now = datetime.now()
            print("#######################################")
            print("########### Starting PCTCE ###########")
            print("########### " + now.strftime("%H:%M:%S" + " ###########"))
            print("#######################################")

        # initialize skeleton
        self.skeleton_initialize()

        # get separation sets
        self.find_sep_set()

        # orientation
        self.rule_origin_causality()

        # test_rp = True
        # test_r2 = True
        # test_r3 = True
        # while test_rp or test_r2 or test_r3:
        #     test_rp = self.rule_propagation_causality()
        #     test_r2 = self.rule_2()
        #     test_r3 = self.rule_3()

        # self.rule_entropy_reduction_gamma()
        # self.rule_entropy_reduction_lambda()

        # self.rule_commun_confounder_and_causal_chain()
        # self.rule_mediator()
        # self.rule_proba_raising_principle()

        # check self causes
        # self.check_self_loops()

        if self.verbose:
            print("######################################")
            print("Final Results (PCTCE)")
            print("######################################")
            print("Extended Summary Graph:")
            print(self.graph.ghat.edges)
        return self.graph.ghat.edges

    def rule_gap_orientation(self):
        """
        gamma heuristic rule from paper
        """
        if self.verbose:
            print("######################################")
            print("Rule gap orientation")
            print("######################################")

        for i in range(self.graph.d):
            for j in range(i + 1, self.graph.d):
                if (self.graph.edges[i, j] == 1) and (self.graph.edges[j, i] == 1):
                    if self.gamma_matrix[self.names[j]].loc[self.names[i]] > 0:
                        if self.verbose:
                            print(str(i) + "-" + str(j) + "g(i,j)>0", end=" ")
                            print("=> orient " + str(i) + "-> " + str(j))
                        self.graph.edges[i, j] = 2
                    if self.gamma_matrix[self.names[j]].loc[self.names[i]] < 0:
                        if self.verbose:
                            print(str(i) + "-" + str(j) + "g(i,j)<0", end=" ")
                            print("=> orient " + str(i) + "<- " + str(j))
                        self.graph.edges[j, i] = 2


if __name__ == "__main__":
    import pandas as pd
    path = "../../../../data/simulated_ts_data/fork/data_"+str(0)+".csv"
    data = pd.read_csv(path, delimiter=',', index_col=0)
    data = data.loc[:1000]

    ci = PCTCE(data, num_processor=1)
    # print(ci.graph.map_names_nodes)
    # print(ci.graph.ghat.edges)
    # print(ci.graph.number_par_all())
    # ci.skeleton_initialize()
    # ci.find_sep_set()
    # print(ci.graph.ghat.edges)
    ci.fit()

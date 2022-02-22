import time
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed

import causal_discovery_class as cd
import networkx as nx

from tools.graph_functions import print_graph, print_temporal_graph, string_nodes


def get_ground_truths(summary_ogt, lags_gt, self_cause):
    nodes = summary_ogt.nodes
    gtrue = nx.DiGraph()
    gtrue.add_nodes_from(nodes)
    sgtrue = nx.DiGraph()
    sgtrue.add_nodes_from(nodes)
    tgtrue = nx.DiGraph()
    tgtrue.add_nodes_from(nodes)

    if self_cause:
        for node in summary_ogt.nodes:
            sgtrue.add_edges_from([(node, node)])
            gtrue.add_edges_from([(node, node)])
            tgtrue.add_edges_from([(node, node)])
            tgtrue.edges[node, node]['time'] = [1]

    for e in summary_ogt.edges:
        # if e[0] == e[1]:
        #     sgtrue.add_edges_from([e])
        #     tgtrue.add_edges_from([e])
        # else:
        gtrue.add_edges_from([e])
        tgtrue.add_edges_from([e])
        tgtrue.edges[e]['time'] = [lags_gt[e]]
    return gtrue, sgtrue, tgtrue

# def get_ground_truth(structure, nodes, selfcause=True):
#     gtrue = nx.DiGraph()
#     gtrue.add_nodes_from(nodes)
#     ogtrue = nx.DiGraph()
#     ogtrue.add_nodes_from(nodes)
#     sgtrue = nx.DiGraph()
#     sgtrue.add_nodes_from(nodes)
#     tgtrue = nx.DiGraph()
#     tgtrue.add_nodes_from(nodes)
#
#     if selfcause:
#         for i in range(len(nodes)):
#             sgtrue.add_edges_from([(nodes[i], nodes[i])])
#             tgtrue.add_edges_from([(nodes[i], nodes[i])])
#
#     if (structure == "fork") or (structure == "fork_0"):
#         if selfcause:
#             # tgtrue.add_edges_from(
#             #     [(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2])])
#             tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
#             tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
#             tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
#
#         ogtrue.add_edges_from([(nodes[0], nodes[1]), (nodes[0], nodes[2])])
#         tgtrue.add_edges_from([(nodes[0], nodes[1]), (nodes[0], nodes[2])])
#         if structure == "fork":
#             tgtrue.edges[nodes[0], nodes[1]]['time'] = [1]
#             tgtrue.edges[nodes[0], nodes[2]]['time'] = [2]
#         else:
#             tgtrue.edges[nodes[0], nodes[1]]['time'] = [0]
#             tgtrue.edges[nodes[0], nodes[2]]['time'] = [0]
#
#     elif (structure == "v_structure") or (structure == "v_structure_0"):
#         if selfcause:
#             # tgtrue.add_edges_from(
#             #     [(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2])])
#             tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
#             tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
#             tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
#
#         ogtrue.add_edges_from([(nodes[0], nodes[2]), (nodes[1], nodes[2])])
#         tgtrue.add_edges_from([(nodes[0], nodes[2]), (nodes[1], nodes[2])])
#         if structure == "v_structure":
#             tgtrue.edges[nodes[0], nodes[2]]['time'] = [1]
#             tgtrue.edges[nodes[1], nodes[2]]['time'] = [2]
#         else:
#             tgtrue.edges[nodes[0], nodes[2]]['time'] = [0]
#             tgtrue.edges[nodes[1], nodes[2]]['time'] = [0]
#     elif (structure == "diamond") or (structure == "diamond_0"):
#         if selfcause:
#             # tgtrue.add_edges_from(
#             #     [(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2]), (nodes[3], nodes[3])])
#             tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
#             tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
#             tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
#             tgtrue.edges[nodes[3], nodes[3]]['time'] = [1]
#
#         ogtrue.add_edges_from([(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], nodes[3]), (nodes[2], nodes[3])])
#         tgtrue.add_edges_from([(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], nodes[3]), (nodes[2], nodes[3])])
#         if structure == "diamond":
#             tgtrue.edges[nodes[0], nodes[1]]['time'] = [1]
#             tgtrue.edges[nodes[0], nodes[2]]['time'] = [2]
#             tgtrue.edges[nodes[1], nodes[3]]['time'] = [2]
#             tgtrue.edges[nodes[2], nodes[3]]['time'] = [1]
#         else:
#             tgtrue.edges[nodes[0], nodes[1]]['time'] = [0]
#             tgtrue.edges[nodes[0], nodes[2]]['time'] = [0]
#             tgtrue.edges[nodes[1], nodes[3]]['time'] = [0]
#             tgtrue.edges[nodes[2], nodes[3]]['time'] = [0]
#
#     gtrue.add_edges_from(ogtrue.edges)
#     gtrue.add_edges_from(sgtrue.edges)
#     return gtrue, ogtrue, sgtrue, tgtrue


def run_on_data(i, method, structure, graph_type, self_cause, max_lag, n_samples, files_input_name, ogtrue, lagstrue,
                verbose):
    save_model = True
    if verbose:
        print("############################## Run "+str(i)+" ##############################")

    file_input_name = files_input_name[i]
    data = pd.read_csv('./data/simulated_ts_data_v2/' + graph_type + '/' + structure + '/datasets/' + file_input_name, index_col=0, header=0, delimiter=',')
    data = data.reset_index(drop=True)
    nodes = string_nodes(data.columns)
    data = data.iloc[:n_samples]
    print(data)
    # ogtrue, sgtrue, tgtrue = get_ground_truth(structure, nodes, self_cause)
    print(ogtrue.edges)
    gtrue, sgtrue, tgtrue = get_ground_truths(ogtrue, lagstrue, self_cause)
    print(gtrue.edges, ogtrue.nodes, sgtrue.nodes)
    print(ogtrue.edges)
    print(sgtrue.edges)

    start = time.time()

    if method == "GrangerPW":
        model = cd.GrangerPW(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "GrangerMV":
        model = cd.GrangerMV(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "GrangerMV2":
        model = cd.GrangerMV2(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "GrangerLasso":
        model = cd.GrangerLasso(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "TCDF":
        model = cd.TCDF(nodes, epochs=1000,  kernel_size=4, dilation_coefficient=4, hidden_layers=1, learning_rate=0.01,
                    sig_level=0.05)
        model.infer_from_data(data)
    elif method == "PCMCICMIknn":
        model = cd.PCMCI(nodes, sig_level=0.05, nlags=max_lag, cond_ind_test="CMIknn")
        model.infer_from_data(data)
    elif method == "PCMCIParCorr":
        model = cd.PCMCI(nodes, sig_level=0.05, nlags=max_lag, cond_ind_test="ParCorr")
        model.infer_from_data(data)
    elif method == "PCMCIplusCMIknn":
        model = cd.PCMCIplus(nodes, sig_level=0.05, nlags=max_lag, cond_ind_test="CMIknn")
        model.infer_from_data(data)
    elif method == "PCMCIplusParCorr":
        model = cd.PCMCIplus(nodes, sig_level=0.05, nlags=max_lag, cond_ind_test="ParCorr")
        model.infer_from_data(data)
    elif method == "oCSE":
        model = cd.OCSE(nodes, sig_level=0.05)
        model.infer_from_data(data)
    elif method == "PCTMI":
        model = cd.PCTMI(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "TPCTMI":
        model = cd.TPCTMI(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "TPCMCITMI":
        model = cd.TPCMCITMI(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "PCTCE":
        model = cd.PCTCE(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "FCITCE":
        model = cd.FCITCE(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "NBCB_pw":
        model = cd.NBCB(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "NBCB":
        model = cd.NBCB(nodes, sig_level=0.05, nlags=max_lag, pairwise=False)
        model.infer_from_data(data)
    elif method == "PWNBCBk":
        model = cd.PWNBCBk(nodes, sig_level=0.05, nlags=max_lag, pairwise=False)
        model.infer_from_data(data)
    elif method == "tsFCI":
        model = cd.TsFCI(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "FCITMI":
        model = cd.FCITMI(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "VarLiNGAM":
        model = cd.VarLiNGAM(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "TiMINO":
        model = cd.TiMINO(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "TsKIKO":
        model = cd.TsKIKO(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    elif method == "Dynotears":
        model = cd.Dynotears(nodes, sig_level=0.05, nlags=max_lag)
        model.infer_from_data(data)
    else:
        model = None
        print("Error: method not found")
        exit(0)

    end = time.time()

    if save_model:
        nx.write_gpickle(model.ghat, "./experiments_v2/graphs/summary_other_and_self_graphs/"+structure+"/"+method + "_" +
                         str(n_samples) + "_" + str(i))
        nx.write_gpickle(model.oghat, "./experiments_v2/graphs/summary_other_graphs/"+structure+"/"+method + "_" +
                         str(n_samples) + "_" + str(i))
        nx.write_gpickle(model.sghat, "./experiments_v2/graphs/summary_self_graphs/"+structure+"/"+method + "_" +
                         str(n_samples) + "_" + str(i))

    # evaluation self and other
    pres_a = model.evaluation(gtrue, evaluation_measure="precision_adjacent")
    rec_a = model.evaluation(gtrue, evaluation_measure="recall_adjacent")
    fscore_a = model.evaluation(gtrue, evaluation_measure="f1_adjacent")
    pres_o = model.evaluation(gtrue, evaluation_measure="precision_oriented")
    rec_o = model.evaluation(gtrue, evaluation_measure="recall_oriented")
    fscore_o = model.evaluation(gtrue, evaluation_measure="f1_oriented")
    # evaluation other
    o_pres_a = model.evaluation(ogtrue, evaluation_measure="other_precision_adjacent")
    o_rec_a = model.evaluation(ogtrue, evaluation_measure="other_recall_adjacent")
    o_fscore_a = model.evaluation(ogtrue, evaluation_measure="other_f1_adjacent")
    o_pres_o = model.evaluation(ogtrue, evaluation_measure="other_precision_oriented")
    o_rec_o = model.evaluation(ogtrue, evaluation_measure="other_recall_oriented")
    o_fscore_o = model.evaluation(ogtrue, evaluation_measure="other_f1_oriented")
    # evaluation self
    s_pres = model.evaluation(sgtrue, evaluation_measure="self_precision")
    s_rec = model.evaluation(sgtrue, evaluation_measure="self_recall")
    s_fscore = model.evaluation(sgtrue, evaluation_measure="self_f1")

    if verbose:
        print('True Graph')
        print_graph(gtrue)
        print('Inferred Graph')
        model.print_graph()

        print("precision adjacent: " + str(pres_a))
        print("recall adjacent: " + str(rec_a))
        print("f-score adjacent: " + str(fscore_a))
        print("precision oriented: " + str(pres_o))
        print("recall oriented: " + str(rec_o))
        print("f-score oriented: " + str(fscore_o))

        print("other precision adjacent: " + str(o_pres_a))
        print("other recall adjacent: " + str(o_rec_a))
        print("other f-score adjacent: " + str(o_fscore_a))
        print("other precision oriented: " + str(o_pres_o))
        print("other recall oriented: " + str(o_rec_o))
        print("other f-score oriented: " + str(o_fscore_o))

        print("self precision: " + str(s_pres))
        print("self recall self: " + str(s_rec))
        print("self f-score self: " + str(s_fscore))

    if not isinstance(model, cd.TemporalGraphicalModel):
        print("Computation time: " + str(end - start))
        return pres_a, rec_a, fscore_a, pres_o, rec_o, fscore_o, o_pres_a, o_rec_a, o_fscore_a, o_pres_o, o_rec_o, \
               o_fscore_o, s_pres, s_rec, s_fscore, (end - start)
    else:
        if save_model:
            nx.write_gpickle(model.tghat, "./experiments_v2/graphs/temporal_graphs/" + structure + "/" + method + "_" +
                             str(n_samples) + "_" + str(i))
        # evaluation temporal
        t_pres = model.temporal_evaluation(tgtrue, evaluation_measure="precision")
        t_rec = model.temporal_evaluation(tgtrue, evaluation_measure="recall")
        t_fscore = model.temporal_evaluation(tgtrue, evaluation_measure="f1")

        if verbose:
            print('True Temporal Graph')
            print_temporal_graph(tgtrue)
            print('Inferred Temporal Graph')
            model.print_temporal_graph()

            print("temporal precision: " + str(t_pres))
            print("temporal recall: " + str(t_rec))
            print("temporal f-score: " + str(t_fscore))

            print("Computation time: "+str(end-start))
        return pres_a, rec_a, fscore_a, pres_o, rec_o, fscore_o, o_pres_a, o_rec_a, o_fscore_a, o_pres_o, o_rec_o, \
               o_fscore_o, s_pres, s_rec, s_fscore, (end - start), t_pres, t_rec, t_fscore


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 5:
        print(len(sys.argv))
        method = sys.argv[1]  # GrangerPW, GrangerMV, TCDF, PCMCICMIknn, PCMCIParCorr, PCTMI, tsFCI, FCITMI VarLiNGAM, TiMINO
        structure = sys.argv[2]
        graph_type = sys.argv[3]
        self_cause = bool(int(sys.argv[4]))  # 1 for True
        max_lag = int(sys.argv[5])
        n_samples = int(sys.argv[6])
        num_processor = int(sys.argv[7])  # -1 for all
        verbose = bool(int(sys.argv[8]))
        print('Argument List:', str(sys.argv))
    else:
        print('Missing arguments so will take default arguments')
        method = "FCITCE"  # GrangerPW, GrangerMV2, GangerLasso, TCDF, PCMCICMIknn, PCMCIParCorr, oCSE, PCTMI, tsFCI, FCITMI, VarLiNGAM, TiMINO, TsKIKO, Dynotears
        structure = "7ts2h_0"
        graph_type = "acyclic"
        self_cause = True
        max_lag = 5
        n_samples = 1000    # 65, 125, 250, 500, 1000, 2000, 4000
        num_processor = 1
        verbose = True
        print('Default Argument List:', str(method), str(structure), n_samples, num_processor)

    path_ground_truth = './data/simulated_ts_data_v2/' + graph_type+ '/' + structure + '/graph.txt'
    param_edge_lags = './data/simulated_ts_data_v2/' + graph_type+ '/' + structure + '/lags.txt'
    gt_summary = nx.read_edgelist(
        path_ground_truth,
        create_using=nx.DiGraph(),
        nodetype=str
    )
    if param_edge_lags is not None:
        edge_lags = dict()
        three_col_format = np.loadtxt(param_edge_lags, dtype=str, delimiter=' ')
        for i in range(three_col_format.shape[0]):
            c = three_col_format[i, 0]
            e = three_col_format[i, 1]
            edge_lags[(c, e)] = int(three_col_format[i, 2])
    else:
        edge_lags = None

    path_input = './data/simulated_ts_data_v2/' + graph_type + '/' + structure + '/datasets'
    if self_cause:
        files_input_name = [f for f in listdir(path_input) if
                            isfile(join(path_input, f)) and not f.startswith('.') and "selfcause=None" in f]
    else:
        files_input_name = [f for f in listdir(path_input) if
                            isfile(join(path_input, f)) and not f.startswith('.') and "selfcause=0" in f]
    print(files_input_name)
    results = Parallel(n_jobs=num_processor)(delayed(run_on_data)(i, method, structure, graph_type, self_cause, max_lag, n_samples,
                                                                  files_input_name, gt_summary, edge_lags, verbose) for i in
                                             range(len(files_input_name)))

    results = np.array(results).reshape(10, -1)
    pres_a_list = results[:, 0]
    rec_a_list = results[:, 1]
    fscore_a_list = results[:, 2]
    pres_o_list = results[:, 3]
    rec_o_list = results[:, 4]
    fscore_o_list = results[:, 5]
    o_pres_a_list = results[:, 6]
    o_rec_a_list = results[:, 7]
    o_fscore_a_list = results[:, 8]
    o_pres_o_list = results[:, 9]
    o_rec_o_list = results[:, 10]
    o_fscore_o_list = results[:, 11]
    s_pres_list = results[:, 12]
    s_rec_list = results[:, 13]
    s_fscore_list = results[:, 14]
    comput_time_list = results[:, 15]

    # method = method+"window=1"
    # method = method+"window=auto_new_lambda"
    # method = method+"window=auto_new_lambda_k0.2"

    result_w_name = str(method)+"_" + str(structure) +"_" + str(graph_type) +"_selfcause="+str(self_cause)+"_"+str(n_samples)

    with open("./experiments_v2/performance_average/summary_other_and_self_performance_average/"+result_w_name, "w+") as file:
        file.write("Precision Adjacent: \n" + str(np.mean(pres_a_list)) + " +- " +
                   str(np.std(pres_a_list)))
        file.write("\n")
        file.write("Recall Adjacent: \n" + str(np.mean(rec_a_list)) + " +- " + str(np.std(rec_a_list)))
        file.write("\n")
        file.write("F-Score Adjacent: \n" + str(np.mean(fscore_a_list)) + " +- " + str(np.std(fscore_a_list)))
        file.write("\n")
        file.write("Precision Oriented: \n" + str(np.mean(pres_o_list)) + " +- " + str(np.std(pres_o_list)))
        file.write("\n")
        file.write("Recall Oriented: \n" + str(np.mean(rec_o_list)) + " +- " + str(np.std(rec_o_list)))
        file.write("\n")
        file.write("F-Score Oriented: \n" + str(np.mean(fscore_o_list)) + " +- " + str(np.std(fscore_o_list)))
        file.write("\n")

        file.write("\n\nComputational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))

    with open("./experiments_v2/performance_average/summary_other_performance_average/"+result_w_name, "w+") as file:
        file.write("Other Precision Adjacent: \n" + str(np.mean(o_pres_a_list)) + " +- " +
                   str(np.std(o_pres_a_list)))
        file.write("\n")
        file.write("Other Recall Adjacent: \n" + str(np.mean(o_rec_a_list)) + " +- " + str(np.std(o_rec_a_list)))
        file.write("\n")
        file.write("Other F-Score Adjacent: \n" + str(np.mean(o_fscore_a_list)) + " +- " + str(np.std(o_fscore_a_list)))
        file.write("\n")
        file.write("Other Precision Oriented: \n" + str(np.mean(o_pres_o_list)) + " +- " + str(np.std(o_pres_o_list)))
        file.write("\n")
        file.write("Other Recall Oriented: \n" + str(np.mean(o_rec_o_list)) + " +- " + str(np.std(o_rec_o_list)))
        file.write("\n")
        file.write("Other F-Score Oriented: \n" + str(np.mean(o_fscore_o_list)) + " +- " + str(np.std(o_fscore_o_list)))
        file.write("\n")

        file.write("\n\nComputational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))

    with open("./experiments_v2/performance_average/summary_self_performance_average/"+result_w_name, "w+") as file:
        file.write("Self Precision: \n" + str(np.mean(s_pres_list)) + " +- " + str(np.std(s_pres_list)))
        file.write("\n")
        file.write("Self Recall: \n" + str(np.mean(s_rec_list)) + " +- " + str(np.std(s_rec_list)))
        file.write("\n")
        file.write("Self F-Score: \n" + str(np.mean(s_fscore_list)) + " +- " + str(np.std(s_fscore_list)))
        file.write("\n")

        file.write("\n\nComputational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))

    if verbose:
        print("####################### Final Result #######################")
        print("Precision Adjacent: " + str(np.mean(pres_a_list)) + " +- " + str(np.std(pres_a_list)))
        print("Recall Adjacent: " + str(np.mean(rec_a_list)) + " +- " + str(np.std(rec_a_list)))
        print("F-Score Adjacent: " + str(np.mean(fscore_a_list)) + " +- " + str(np.std(fscore_a_list)))
        print("Precision Oriented: " + str(np.mean(pres_o_list)) + " +- " + str(np.std(pres_o_list)))
        print("Recall Oriented: " + str(np.mean(rec_o_list)) + " +- " + str(np.std(rec_o_list)))
        print("F-Score Oriented: " + str(np.mean(fscore_o_list)) + " +- " + str(np.std(fscore_o_list)))
        print("Other Precision Adjacent: " + str(np.mean(o_pres_a_list)) + " +- " + str(np.std(o_pres_a_list)))
        print("Other Recall Adjacent: " + str(np.mean(o_rec_a_list)) + " +- " + str(np.std(o_rec_a_list)))
        print("Other F-Score Adjacent: " + str(np.mean(o_fscore_a_list)) + " +- " + str(np.std(o_fscore_a_list)))
        print("Other Precision Oriented: " + str(np.mean(o_pres_o_list)) + " +- " + str(np.std(o_pres_o_list)))
        print("Other Recall Oriented: " + str(np.mean(o_rec_o_list)) + " +- " + str(np.std(o_rec_o_list)))
        print("Other F-Score Oriented: " + str(np.mean(o_fscore_o_list)) + " +- " + str(np.std(o_fscore_o_list)))
        print("Self Precision: " + str(np.mean(s_pres_list)) + " +- " + str(np.std(s_pres_list)))
        print("Self Recall: " + str(np.mean(s_rec_list)) + " +- " + str(np.std(s_rec_list)))
        print("Self F-Score: " + str(np.mean(s_fscore_list)) + " +- " + str(np.std(s_fscore_list)))
        print("Computational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))

    if results.shape[1] > 16:
        t_pres_list = results[:, 16]
        t_rec_list = results[:, 17]
        t_fscore_list = results[:, 18]

        with open("./experiments_v2/performance_average/temporal_performance_average/" + result_w_name, "w+") as file:
            file.write("Temporal Precision: \n" + str(np.mean(t_pres_list)) + " +- " + str(np.std(t_pres_list)))
            file.write("\n")
            file.write("Temporal Recall: \n" + str(np.mean(t_rec_list)) + " +- " + str(np.std(t_rec_list)))
            file.write("\n")
            file.write("Temporal F-Score: \n" + str(np.mean(t_fscore_list)) + " +- " + str(np.std(t_fscore_list)))
            file.write("\n")

            file.write("\n\nComputational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))
        if verbose:
            print("Temporal Precision: " + str(np.mean(t_pres_list)) + " +- " + str(np.std(t_pres_list)))
            print("Temporal Recall: " + str(np.mean(t_rec_list)) + " +- " + str(np.std(t_rec_list)))
            print("Temporal F-Score: " + str(np.mean(t_fscore_list)) + " +- " + str(np.std(t_fscore_list)))
        np.savetxt("./experiments_v2/performance_detail/" + result_w_name +
                   ".csv", results, delimiter=';',
                   header="pres_a, rec_a, fscore_a, pres_o, rec_o, fscore_o, o_pres_a, o_rec_a, o_fscore_a, o_pres_o, "
                          "o_rec_o, o_fscore_o, s_pres, s_rec, s_fscore, computational_time, t_pres, t_rec, t_fscore")
    else:
        np.savetxt("./experiments_v2/performance_detail/" + result_w_name +
                   ".csv", results, delimiter=';',
                   header="pres_a, rec_a, fscore_a, pres_o, rec_o, fscore_o, o_pres_a, o_rec_a, o_fscore_a, o_pres_o, "
                          "o_rec_o, o_fscore_o, s_pres, s_rec, s_fscore, computational_time")


import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, \
    precision_score, f1_score, precision_recall_curve, accuracy_score, balanced_accuracy_score

# -------------------------------------methods called-----------------------------------------------

from elp_all_link_pred_algo import aa, ra, cclp, cclp2, cn, pa, jc, car, \
    rooted_pagerank_linkpred, normalize, clp_id_main, elp

import time
import random
from xlwt import Workbook
import xlrd
import datetime
import os
import shutil
import subprocess
import sys
from psutil import virtual_memory
from scipy.stats import friedmanchisquare
import scikit_posthocs as sci_post

from gml_old_prev import read_gml_old
from networkx import read_gml

if __name__ == '__main__':
    starttime_full = time.time()
    var_dict_main = {}

    def auprgraph_all (adj,file_name,algo):
        print("for algo - "+str(algo))
        file_write_name = './result_all_elp/result_'+algo+'/' + file_name + ".txt"
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        starttime_aup = time.time()
        ratio = []
        aupr = []
        recall = []
        auc = []
        avg_prec = []
        acc_score = []
        bal_acc_score = []
        f1 = []
        prec = []
        G = nx.Graph(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        print("nodes - " + str(len(adj)) + " edges - " + str(G.number_of_edges()) + " name - " + str(file_name))
        for i in [0.5,0.6,0.7,0.8,0.9]: # range is the fraction of edge values included in the graph
            print("nodes - " + str(len(adj)) + " edges - " + str(G.number_of_edges()) + " name - " + str(file_name))
            print ("For ratio : " , i-1)
            avg_array = avg_seq_all(G, file_name, i, algo)
            aupr.append(avg_array[0])
            recall.append(avg_array[1])
            auc.append(avg_array[2])
            avg_prec.append(avg_array[3])
            acc_score.append(avg_array[4])
            bal_acc_score.append(avg_array[5])
            f1.append(avg_array[6])
            prec.append(avg_array[7])
            ratio.append(i-1)
        print("Ratio:-", ratio)
        print("AUPR:-",aupr)
        print("Recall:-",recall)
        print("AUC:-",auc)
        print("Avg Precision:-",avg_prec)
        print("Accuracy Score:-",acc_score)
        print("Balanced Accuracy Score:-", bal_acc_score)
        print("F1 Score:-", f1)
        print("Precision Score:-", prec)
        endtime_aup = time.time()
        print('That aup took {} seconds'.format(endtime_aup - starttime_aup))

        # Workbook is created
        wb = Workbook()
        # add_sheet is used to create sheet.
        sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)
        sheet1.write(0, 0, 'Ratio')
        sheet1.write(0, 1, 'AUPR')
        sheet1.write(0, 2, 'RECALL')
        sheet1.write(0, 3, 'AUC')
        sheet1.write(0, 4, 'AVG PRECISION')
        sheet1.write(0, 5, 'ACCURACY SCORE')
        sheet1.write(0, 6, 'BAL ACCURACY SCORE')
        sheet1.write(0, 7, 'F1 MEASURE')
        sheet1.write(0, 8, 'PRECISION')
        for i in range(5):
            sheet1.write(5 - i, 0, ratio[i]*-1)
            sheet1.write(5 - i, 1, aupr[i])
            sheet1.write(5 - i, 2, recall[i])
            sheet1.write(5 - i, 3, auc[i])
            sheet1.write(5 - i, 4, avg_prec[i])
            sheet1.write(5 - i, 5, acc_score[i])
            sheet1.write(5 - i, 6, bal_acc_score[i])
            sheet1.write(5 - i, 7, f1[i])
            sheet1.write(5 - i, 8, prec[i])

        wb.save('./result_all_elp/result_'+algo+'/' + file_name + ".xls")

        currentDT = datetime.datetime.now()
        print(str(currentDT))

        file_all = open('./result_all_elp/current_all.txt','a')
        text_final = "full algo = "+algo+" file name = "+file_name+" time = "+\
                     str((endtime_aup - starttime_aup))+" date_time = "+str(currentDT)+"\n"
        file_all.write(text_final)
        print(text_final)
        file_all.close()

        return aupr,ratio,recall,auc,avg_prec,acc_score,bal_acc_score,f1,prec


    def avg_seq_all(g, file_name, ratio, algo) :

        start_time_ratio = time.time()
        aupr = 0
        recall = 0
        auc = 0
        avg_prec = 0
        acc_score = 0
        bal_acc_score = 0
        f1 = 0
        prec = 0
        loop = 5
        ratio = round(ratio, 1)
        graph_original = g
        print("avg sequential called for algo - " + str(algo) + " ratio - " + str(ratio))

        for single_iter in range(loop):

            print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
            # making original graph adjacency matrix
            adj_original = nx.adjacency_matrix(graph_original).todense()
            starttime = time.time()
            # finding edges and nodes of original graph
            edges = np.array(list(graph_original.edges))
            nodes = list(range(len(adj_original)))
            np.random.shuffle(edges)
            edges_original = edges
            edges_train = np.array(edges_original, copy=True)
            np.random.shuffle(edges_train)
            edges_train = random.sample(list(edges_train), int(ratio * (len(edges_train))))
            # finding training set of edges according to ratio
            graph_train = nx.Graph()
            # making graph based on the training edges
            graph_train.add_nodes_from(nodes)
            graph_train.add_edges_from(edges_train)
            adj_train = nx.adjacency_matrix(graph_train).todense()
            # making test graph by removing train edges from original
            graph_test = nx.Graph()
            graph_test.add_nodes_from(nodes)
            graph_test.add_edges_from(edges_original)
            graph_test.remove_edges_from(edges_train)
            print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))

            # sending training graph for probability matrix prediction
            if algo == 'cn': prob_mat = cn(adj_train)
            if algo == 'ra': prob_mat = ra(adj_train)
            if algo == 'car': prob_mat = car(adj_train)
            if algo == 'cclp': prob_mat = cclp(adj_train)
            if algo == 'cclp2': prob_mat = cclp2(adj_train)
            if algo == 'jc': prob_mat = jc(adj_train)
            if algo == 'pa': prob_mat = pa(adj_train)
            if algo == 'pagerank': prob_mat = rooted_pagerank_linkpred(adj_train)
            if algo == 'aa': prob_mat = aa(adj_train)
            if algo == 'elp': prob_mat = elp(adj_train)
            if algo == 'clp_id': prob_mat = clp_id_main(adj_train, 25, 1.0)

            prob_mat = normalize(prob_mat)
            endtime = time.time()
            print('{} for probability matrix prediction'.format(endtime - starttime))

            # making adcancecy test from testing graph
            adj_test = nx.adjacency_matrix(graph_test).todense()
            # making new arrays to pass to function
            array_true = []
            array_pred = []
            for i in range(len(adj_original)):
                for j in range(len(adj_original)):
                    if not graph_original.has_edge(i, j):
                        array_true.append(0)
                        array_pred.append(prob_mat[i][j])
                    if graph_test.has_edge(i, j):
                        array_true.append(1)
                        array_pred.append(prob_mat[i][j])
            # flattening adjacency matrices
            '''pred = pred.flatten()
            adj_original = np.array(adj_original).flatten()
            adj_test = np.array(adj_test).flatten()'''
            pred = array_pred
            adj_test = array_true

            # return precision recall pairs for particular thresholds
            prec_per, recall_per, threshold_per = precision_recall_curve(adj_test, pred)
            prec_per = prec_per[::-1]
            recall_per = recall_per[::-1]
            aupr_value = np.trapz(prec_per, x=recall_per)
            auc_value = roc_auc_score(adj_test, pred)
            avg_prec_value = average_precision_score(adj_test, pred)

            test_pred_label = np.copy(pred)
            a = np.mean(test_pred_label)

            for i in range(len(pred)):
                if pred[i] < a:
                    test_pred_label[i] = 0
                else:
                    test_pred_label[i] = 1
            recall_value = recall_score(adj_test, test_pred_label)
            acc_score_value = accuracy_score(adj_test, test_pred_label)
            bal_acc_score_value = balanced_accuracy_score(adj_test, test_pred_label)
            precision_value = precision_score(adj_test, test_pred_label)
            f1_value = f1_score(adj_test, test_pred_label)

            endtime = time.time()
            print('{} for metric calculation'.format(endtime - starttime))

            currentDT = datetime.datetime.now()
            print(str(currentDT))

            file_all = open('./result_all_elp/current.txt', 'a')
            text_inside_single = "single algo = " + algo + " file name = " + file_name + \
                                 " ratio = " + str(ratio) + " time = " + \
                                 str(endtime - starttime) + " sec date_time = " + str(currentDT) + "\n"
            file_all.write(text_inside_single)
            print(text_inside_single)
            file_all.close()

            aupr += aupr_value
            recall += recall_value
            auc += auc_value
            avg_prec += avg_prec_value
            acc_score += acc_score_value
            bal_acc_score += bal_acc_score_value
            f1 += f1_value
            prec += precision_value

        currentDT = datetime.datetime.now()
        print(str(currentDT))
        end_time_ratio = time.time()
        file_all = open('./result_all_elp/current.txt', 'a')
        text_inside = "full algo = " + algo + " file name = " + file_name + \
                           " ratio = " + str(ratio) + " time = " + \
                           str(end_time_ratio - start_time_ratio) + " date_time = " \
                           + str(currentDT) + "\n"
        file_all.write(text_inside)
        file_all.close()

        return aupr / loop, recall / loop, auc / loop, avg_prec / loop, acc_score / loop, \
               bal_acc_score / loop, f1 / loop, prec / loop


    def result_parser_combine(file_name_array,algo_result_all_metric):

        algo_all = algo_result_all_metric

        for file_name in file_name_array:
            file_write_name = './result_all_elp/' + file_name + "_combine.xls"
            os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
            # Workbook is created
            wb_write = Workbook()
            # add_sheet is used to create sheet.
            AUPR = wb_write.add_sheet('AUPR', cell_overwrite_ok=True)
            RECALL = wb_write.add_sheet('RECALL', cell_overwrite_ok=True)
            AUC = wb_write.add_sheet('AUC', cell_overwrite_ok=True)
            AVG_PREC = wb_write.add_sheet('AVG PREC', cell_overwrite_ok=True)
            ACC_SCORE = wb_write.add_sheet('ACC SCORE', cell_overwrite_ok=True)
            BAL_ACC_SCORE = wb_write.add_sheet('BAL ACC SCORE', cell_overwrite_ok=True)
            F1_SCORE = wb_write.add_sheet('F1 SCORE', cell_overwrite_ok=True)
            PRECISION = wb_write.add_sheet('PRECISION', cell_overwrite_ok=True)
            sheet_array = [AUPR, RECALL, AUC, AVG_PREC, ACC_SCORE, BAL_ACC_SCORE, F1_SCORE, PRECISION]
            for sheet_single in sheet_array:
                sheet_single.write(0, 0, 'Ratio')
                sheet_single.write(1, 0, '0.1')
                sheet_single.write(2, 0, '0.2')
                sheet_single.write(3, 0, '0.3')
                sheet_single.write(4, 0, '0.4')
                sheet_single.write(5, 0, '0.5')
            current_algo = 1
            for algo in algo_all:
                single_algo_file = "./result_all_elp/result_" + str(algo) + '/' + file_name + ".xls"
                wb_read = xlrd.open_workbook(single_algo_file)
                main_sheet = wb_read.sheet_by_name('Sheet 1')
                for sheet_single in sheet_array:
                    sheet_single.write(0, current_algo, str(algo).upper())
                for row_read in range(5):
                    row_read += 1
                    row_write = row_read
                    for col_read in range(8):
                        col_read += 1
                        print("reading--" + file_name + " --of algo--" + algo)
                        value = float(main_sheet.cell(row_read, col_read).value)
                        value = round(value, 5)
                        sheet_no = col_read - 1
                        sheet_array[sheet_no].write(row_write, current_algo, value)
                current_algo = current_algo + 1
            wb_write.save(file_write_name)

        wb_dataset_write = Workbook()
        file_dataset_write_name = './result_all_elp/all_datasets_combine_elp.xls'
        sheet_name_array = ['AUPR', 'RECALL', 'AUC', 'AVG PREC', 'ACC SCORE', 'BAL ACC SCORE',
                            'F1 SCORE', 'PRECISION']
        AUPR_write = wb_dataset_write.add_sheet(sheet_name_array[0], cell_overwrite_ok=True)
        RECALL_write = wb_dataset_write.add_sheet(sheet_name_array[1], cell_overwrite_ok=True)
        AUC_write = wb_dataset_write.add_sheet(sheet_name_array[2], cell_overwrite_ok=True)
        AVG_PREC_write = wb_dataset_write.add_sheet(sheet_name_array[3], cell_overwrite_ok=True)
        ACC_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[4], cell_overwrite_ok=True)
        BAL_ACC_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[5], cell_overwrite_ok=True)
        F1_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[6], cell_overwrite_ok=True)
        PRECISION_write = wb_dataset_write.add_sheet(sheet_name_array[7], cell_overwrite_ok=True)
        sheet_dataset_write_array = [AUPR_write, RECALL_write, AUC_write, AVG_PREC_write, ACC_SCORE_write,
                                     BAL_ACC_SCORE_write, F1_SCORE_write, PRECISION_write]
        count = 0
        for file_name in file_name_array:
            file_read_name = './result_all_elp/' + file_name + "_combine.xls"
            wb_read = xlrd.open_workbook(file_read_name)
            AUPR = wb_read.sheet_by_name(sheet_name_array[0])
            RECALL = wb_read.sheet_by_name(sheet_name_array[1])
            AUC = wb_read.sheet_by_name(sheet_name_array[2])
            AVG_PREC = wb_read.sheet_by_name(sheet_name_array[3])
            ACC_SCORE = wb_read.sheet_by_name(sheet_name_array[4])
            BAL_ACC_SCORE = wb_read.sheet_by_name(sheet_name_array[5])
            F1_SCORE = wb_read.sheet_by_name(sheet_name_array[6])
            PRECISION_SCORE = wb_read.sheet_by_name(sheet_name_array[7])
            sheet_read_array = [AUPR, RECALL, AUC, AVG_PREC, ACC_SCORE, BAL_ACC_SCORE, F1_SCORE, PRECISION_SCORE]
            write_row = file_name_array.index(file_name) + 1
            for sheet_no in range(len(sheet_read_array)):
                sheet_dataset_write_array[sheet_no].write(0, 1, 'Ratio')
                sheet_dataset_write_array[sheet_no].write(1 + count * 6, 1, '0.1')
                sheet_dataset_write_array[sheet_no].write(2 + count * 6, 1, '0.2')
                sheet_dataset_write_array[sheet_no].write(3 + count * 6, 1, '0.3')
                sheet_dataset_write_array[sheet_no].write(4 + count * 6, 1, '0.4')
                sheet_dataset_write_array[sheet_no].write(5 + count * 6, 1, '0.5')
                sheet_dataset_write_array[sheet_no].write(0, 0, 'FILE_NAME')
                sheet_dataset_write_array[sheet_no].write(1 + count * 6, 0, str(file_name))
                for ratio in range(5):
                    read_row = ratio + 1
                    write_row = ratio + 1 + count * 6
                    for algo_no in range(len(algo_all)):
                        read_col = algo_no + 1
                        write_col = algo_no + 2
                        value = sheet_read_array[sheet_no].cell(read_row, read_col).value
                        print("value read = " + str(value))
                        # sheet_dataset_write_array[sheet_no].write(count * 6, write_col, str(algo_all[algo_no]).upper())
                        sheet_dataset_write_array[sheet_no].write(0, write_col, str(algo_all[algo_no]).upper())
                        sheet_dataset_write_array[sheet_no].write(write_row, write_col, value)
            count += 1
        wb_dataset_write.save(file_dataset_write_name)


    def aupgraph_control_multiple_dataset_all(file_name_array=['fb_U', 'twitter_U']):
        file_write_name = './result_all_elp/current_all.txt'
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        algo_ego = ['cn', 'ra', 'car', 'cclp', 'jc', 'pa', 'pagerank',
                    'clp_id', 'elp']
        for algo in algo_ego :
            if algo not in [] :
                for file_name in file_name_array:
                    ds = './datasets/' + file_name
                    if file_name == "polblogs":
                        h = read_gml(ds + '.gml')
                    else:
                        h = read_gml_old(ds + '.gml')
                    adj_mat_s = nx.adjacency_matrix(h)
                    n = adj_mat_s.shape[0]
                    print("nodes = "+str(n))
                    adj_mat_d = adj_mat_s.todense()
                    adj = adj_mat_d
                    auprgraph_all(adj, file_name, algo)


    algo_result_ego = ['cn', 'ra', 'car', 'cclp', 'jc', 'pa', 'pagerank',
                        'clp_id', 'elp']

    file_name_array = ['jazz', 'karate', 'celegansneural', 'football', 'dolphin', 'usair97',
                       'airlines', 'polblogs', 'netscience', 'PB_U', 'SmaGri_U', 'NS_U',
                       'online_soc_net', 'celegans_metabolic', 'email.Eu.core', 'amazon',
                       'cora', 'PPI_U', 'yeast1', 'GrQc_U', 'power', 'fb_U', 'twitter_U', 'Router']
    file_name_array_ego = ['karate', 'jazz', 'celegansneural', 'polblogs', 'SmaGri_U', 'GrQc_U']

    aupgraph_control_multiple_dataset_all()
    #result_parser_combine(file_name_array_ego,algo_result_ego)

    print('That took {} seconds'.format(time.time() - starttime_full))



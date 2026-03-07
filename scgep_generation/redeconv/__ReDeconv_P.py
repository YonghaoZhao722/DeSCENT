# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:45:43 2023

@author: Songjian Lu
"""
import scipy.stats as stats
import numpy, math, time, random
from numpy.linalg import inv
from scipy.optimize import nnls
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  


p_st_development = 'N'
p_st_using = 'Y'

#============================================================================================

def read_meta_info(p_file, p_cid_colNo = 0, p_type_colNo = 1, p_sp = '\t'):
   L_rets = {}; L_rets2 = {}
   ff = open(p_file, 'r')
   count = 0
   for ln in ff:
      count += 1
      if count>1:
         ln0 = ln.split('\n')
         ln1 = ln0[0].split(p_sp)
         ctype = ln1[p_type_colNo]
         cid = ln1[p_cid_colNo]
         if not ctype in L_rets:
            L_rets[ctype] = {}
         L_rets[ctype][cid] = 1
         ctype2 = ctype.split('"')
         if len(ctype2)>1:
            ctype = ctype2[1]
         L_rets2[cid] = ctype

   ff.close()

   if (p_st_using.upper())[0] == 'Y':  
      print('--- Counts of different cell types ---')
   for tp in L_rets:
      if (p_st_using.upper())[0] == 'Y':
         #print(tp, '\t', len(L_rets[tp]))
         print(' +++',tp, '\t', len(L_rets[tp]))

   count = 0
   for item in L_rets2:
      count += 1
      if count<10:
         print(item, L_rets2[item])
         
   return L_rets, L_rets2


def read_meta_info_2(p_file, p_CellNo_LB=30, p_cid_colNo = 0, p_type_colNo = 1, p_sp = '\t'):
   L_rets = {}; L_rets2 = {}
   ff = open(p_file, 'r')
   count = 0
   for ln in ff:
      count += 1
      if count>1:
         ln0 = ln.split('\n')
         ln1 = ln0[0].split(p_sp)
         ctype = ln1[p_type_colNo]
         cid = ln1[p_cid_colNo]
         if not ctype in L_rets:
            L_rets[ctype] = {}
         L_rets[ctype][cid] = 1
         ctype2 = ctype.split('"')
         if len(ctype2)>1:
            ctype = ctype2[1]
         L_rets2[cid] = ctype

   ff.close()
   
   L_rets_new = {}
   for ctp in L_rets:
      if len(L_rets[ctp])>=p_CellNo_LB:
         L_rets_new[ctp] = L_rets[ctp]

   if (p_st_using.upper())[0] == 'Y':  
      print('--- Counts of different cell types ---')
   for tp in L_rets:
      if len(L_rets[tp])>=p_CellNo_LB:
         #print(tp, '\t', len(L_rets[tp]))
         print(' +++',tp, '\t', len(L_rets[tp]))
      else:
         print(' ---',tp, '\t', len(L_rets[tp]), '\tExcluded!')

   count = 0
   for item in L_rets2:
      count += 1
      if count<5:
         print(item, L_rets2[item])
         
   return L_rets_new, L_rets2

def get_initial_Signature_genes_new_2(p_exp, p_groups, p_out, p_pv=0.05, p_fd = 2.0, p_level = 3, p_grp_geneNo = 1000, p_sp_in = '\t', p_sp_out = '\t'):
   # A gene that is considered as a signature gene must be significant  
   # up-regulated in one group (Comparing with all other groups)
   L_rets = {}; L_rets2 = {}; L_grps = {}; L_grp_list = []
   for grp in p_groups:
      L_rets[grp] = {}
      L_rets2[grp] = {}

   ff = open(p_exp, 'r')
   count = 0
   for ln in ff:
      count += 1
      ln1 = ln.split('\n')
      ln2 = ln1[0].split(p_sp_in)
      if count == 1:
         for idx in range(len(ln2)):
            item = ln2[idx]
            for grp in p_groups:
               if item in p_groups[grp]:
                  if not grp in L_grps:
                     L_grps[grp] = []
                  L_grps[grp].append(idx)
         t_total = 0
         for grp in L_grps:
            if (p_st_using.upper())[0] == 'Y':
               print('--Grp:', grp, len(L_grps[grp]))
            t_total += len(L_grps[grp])
            L_grp_list.append(grp)
         if (p_st_using.upper())[0] == 'Y':
            print('---Total sample:', t_total)

      else:
         if count % 100 == 0:
            if (p_st_using.upper())[0] == 'Y':
               print('--- Processing row: ',count, ' gene: ',ln2[0])
         gene = ln2[0]
         t_grp_exp = {}
         for grp in L_grp_list:
            t_grp_exp[grp] = []
            for idx in L_grps[grp]:
               t_grp_exp[grp].append(float(ln2[idx]))
         for grp in L_grp_list:
            t_exp1 = t_grp_exp[grp]
            t_st = 0; t_max_pv = 0; t_min_fd = math.inf; t_mid = 0; t_m2_zero = 0
            tmean = numpy.mean(t_exp1)
            tstd = numpy.std(t_exp1, axis=0)            
            for grp2 in L_grp_list:
               if not grp2==grp:
                  t_exp2 = t_grp_exp[grp2]
                  t_stat1_2, p_val1_2 = stats.ttest_ind(t_exp1, t_exp2, equal_var=False)

                  tmean2 = numpy.mean(t_exp2)

                  if p_val1_2<=p_pv:
                     if p_val1_2>t_max_pv:
                        t_max_pv = p_val1_2
                     if tmean2>0:
                        if tmean/tmean2>=p_fd:
                           t_st += 1
                        if t_min_fd>tmean/tmean2:
                           t_min_fd = tmean/tmean2
                     else:
                        if tmean>0:
                           t_st += 1
                           t_m2_zero += 1
            t_level = len(p_groups) - t_st
            #if t_st >= len(p_groups)-1:
            if t_level <= p_level:
               if not t_level in L_rets[grp]:
                  L_rets[grp][t_level] = {}
               L_rets[grp][t_level][gene] = [str(t_max_pv), str(t_min_fd), str(t_level), str(tmean), str(tstd)]

            
   ff.close()
   if (p_st_using.upper())[0] == 'Y':
      print('\n--- new signature genes:')
   L_total_genes = {}
   for grp in L_grp_list:
      if (p_st_using.upper())[0] == 'Y':
         print(grp, len(L_rets[grp]))
      for tlv in L_rets[grp]:
         if (p_st_using.upper())[0] == 'Y':
            print('   ',tlv, len(L_rets[grp][tlv]))
         # count = 0
         # for gene in L_rets[grp][tlv]:
         #    L_total_genes[gene] = 1
         #    count += 1
         #    if count<=2:
         #       print('   --',tlv,gene,L_rets[grp][tlv][gene])
         
   if (p_st_using.upper())[0] == 'Y':
      print('----------------')
      
      print('---Total signature#:', len(L_total_genes))        
   fout = open(p_out, 'w')
   for grp in L_rets:
      t_count_grp = 0
      t_lv_all = []
      for tlv in L_rets[grp]:
         t_lv_all.append(tlv)
      t_lv_all.sort()
      for tlv in t_lv_all:
         if t_count_grp<=p_grp_geneNo:
            for gene in L_rets[grp][tlv]:
               output = gene + p_sp_out + grp + p_sp_out + p_sp_out.join(L_rets[grp][tlv][gene])+'\n'
               fout.write(output)
            t_count_grp += len(L_rets[grp][tlv])
         else:
            break

   fout.close()
   return L_total_genes

def get_initial_Signature_genes_new_3(p_exp, p_groups, p_out, p_pv=0.05, p_fd = 2.0, p_level = 2, p_grp_geneNo = 1000, p_sp_in = '\t', p_sp_out = '\t'):
   # A gene that is considered as a signature gene must be significant  
   # up-regulated in one group (Comparing with all other groups)
   # expression total accross all cells should be larger than mean(all totals)-std
   
   Level_UB = min(len(p_groups) - 2, p_level)
   
   
   L_rets = {}; L_rets2 = {}; L_grps = {}; L_grp_list = []; L_exp_sum = {}
   for grp in p_groups:
      L_rets[grp] = {}
      L_rets2[grp] = {}

   ff = open(p_exp, 'r')
   count = 0
   for ln in ff:
      count += 1
      ln1 = ln.split('\n')
      ln2 = ln1[0].split(p_sp_in)
      if count == 1:
         for idx in range(len(ln2)):
            item = ln2[idx]
            for grp in p_groups:
               if item in p_groups[grp]:
                  if not grp in L_grps:
                     L_grps[grp] = []
                  L_grps[grp].append(idx)
         t_total = 0
         for grp in L_grps:
            if (p_st_using.upper())[0] == 'Y':
               print('--Grp:', grp, len(L_grps[grp]))
            t_total += len(L_grps[grp])
            L_grp_list.append(grp)
         if (p_st_using.upper())[0] == 'Y':
            print('---Total sample:', t_total)

      else:
         if count % 100 == 0:
            if (p_st_using.upper())[0] == 'Y':
               print('--- Processing row: ',count, ' gene: ',ln2[0])
         gene = ln2[0]
         L_exp_sum[gene] = 0.0
         t_grp_exp = {}
         for grp in L_grp_list:
            t_grp_exp[grp] = []
            for idx in L_grps[grp]:
               t_va = float(ln2[idx])
               t_grp_exp[grp].append(t_va)
               L_exp_sum[gene] += t_va
         for grp in L_grp_list:
            t_exp1 = t_grp_exp[grp]
            t_st = 0; t_max_pv = 0; t_min_fd = math.inf; t_mid = 0; t_m2_zero = 0
            tmean = numpy.mean(t_exp1)
            tstd = numpy.std(t_exp1, axis=0)
            t_ctp_missed = []
            for grp2 in L_grp_list:
               if not grp2==grp:
                  t_exp2 = t_grp_exp[grp2]
                  t_stat1_2, p_val1_2 = stats.ttest_ind(t_exp1, t_exp2, equal_var=False)

                  tmean2 = numpy.mean(t_exp2)

                  if p_val1_2<=p_pv:
                     if p_val1_2>t_max_pv:
                        t_max_pv = p_val1_2
                     if tmean2>0:
                        if tmean/tmean2>=p_fd:
                           t_st += 1
                        else:
                           t_ctp_missed.append(grp2)
                        if t_min_fd>tmean/tmean2:
                           t_min_fd = tmean/tmean2
                     else:
                        if tmean>0:
                           t_st += 1
                           t_m2_zero += 1
                        else:
                           t_ctp_missed.append(grp2)
                  else:
                     t_ctp_missed.append(grp2)
                     if p_val1_2>t_max_pv:
                        t_max_pv = p_val1_2                     
                     if tmean2>0:
                        if t_min_fd>tmean/tmean2:
                           t_min_fd = tmean/tmean2                     
                     
            t_level = len(p_groups) - t_st - 1
            #if t_st >= len(p_groups)-1:
            if t_level <= Level_UB:
               if not t_level in L_rets[grp]:
                  L_rets[grp][t_level] = {}
               L_rets[grp][t_level][gene] = [str(t_max_pv), str(t_min_fd), str(t_level), str(tmean), str(tstd), '|'.join(t_ctp_missed)]

            
   ff.close()
   if (p_st_using.upper())[0] == 'Y':
      print('\n--- new signature genes:')
   L_total_genes = {}
   for grp in L_grp_list:
      if (p_st_using.upper())[0] == 'Y':
         print(grp, len(L_rets[grp]))
      tp_level_list = []
      for tlv in L_rets[grp]:
         tp_level_list.append(tlv)
      tp_level_list.sort()
      for tlv in tp_level_list:
         if (p_st_using.upper())[0] == 'Y':
            print('   ',tlv, len(L_rets[grp][tlv]))
         # count = 0
         for gene in L_rets[grp][tlv]:
             L_total_genes[gene] = 1
         #    count += 1
         #    if count<=2:
         #       print('   --',tlv,gene,L_rets[grp][tlv][gene])
   if (p_st_using.upper())[0] == 'Y':
      print('----------------')
   
   L_sum_all = []
   for gene in L_exp_sum:
      L_sum_all.append(L_exp_sum[gene])
   t_mean = numpy.mean(L_sum_all)
   t_std = numpy.std(L_sum_all, axis=0) 
   t_threshold_exp_sum = t_mean - t_std
   
   if (p_st_using.upper())[0] == 'Y':
      print('---Total signature#:', len(L_total_genes))        
   fout = open(p_out, 'w')
   for grp in L_rets:
      t_count_grp = 0
      t_lv_all = []
      for tlv in L_rets[grp]:
         t_lv_all.append(tlv)
      t_lv_all.sort()
      for tlv in t_lv_all:
         if t_count_grp<=p_grp_geneNo:
            t_count_in = 0
            for gene in L_rets[grp][tlv]:
               if L_exp_sum[gene]>= t_threshold_exp_sum:
                  output = gene + p_sp_out + grp + p_sp_out + p_sp_out.join(L_rets[grp][tlv][gene])+'\n'
                  fout.write(output)
                  t_count_in += 1
            #t_count_grp += len(L_rets[grp][tlv])
            t_count_grp += t_count_in
         else:
            break

   fout.close()
   return L_total_genes

def check_meta_and_scRNAseq_data(p_meta, p_exp):
   L_rets = 1
   
   ff = open(p_exp, 'r')
   L_title = []
   count = 0
   for ln in ff:
      ln1 = ln.split('\n')
      ln2 = ln1[0].split('\t')
      count += 1
      if count == 1:
         L_title = ln2
      else:
         if len(ln2)==len(L_title):
            break
         L_rets = -1
         output = '\n !!! In the expression file, the numbers of columns in the first and second rows are not equal ---'
         output += '\n   --- Number of columns in the fist row: ' + str(len(L_title))
         output += '\n   --- Number of columns in the second row: ' + str(len(ln2))
         output += '\n !!! Please check the input data files before run the program --- '
                  

         print(output)
         break
   ff.close()
   if L_rets<0:
      return L_rets
   
   L_cell_all = {}; L_cell_list = []
   ff = open(p_meta, 'r')
   count = 0
   for ln in ff:
      count += 1
      if count>1:
         ln1 = ln.split('\t')
         L_cell_all[ln1[0]] = 1
         L_cell_list.append(ln1[0])
   ff.close()
   if len(L_cell_list)+1 == len(L_title):
      t_matched = 0
      for idx in range(1, len(L_title)):
         if L_title[idx] in L_cell_all:
            t_matched += 1
      if t_matched<len(L_cell_all):
         L_rets = -1
         output = '\n !!! Some cell IDs in the meta and expression data files are not matched ---'
         output += '\n   --- Number of cells in the expression data file: ' + str(len(L_title)-1)
         output += '\n   --- Number of cells in the meta data file: ' + str(len(L_cell_all))
         output += '\n   --- Number of cells mathced in both files: ' + str(t_matched)
         output += '\n !!! Please check the input data files before run the program --- '
                  

         print(output)         
   else:
         L_rets = -1
         output = '\n !!! The numbers of cells in the meta and expression data files are not equal ---'
         output += '\n   --- Number of cells in the expression data file: ' + str(len(L_title)-1)
         output += '\n   --- Number of cells in the meta data file: ' + str(len(L_cell_list))
         output += '\n !!! Please check the input data files before run the program --- '
                  
         print(output)
   return L_rets

def get_initial_Signature_Candidates(p_meta, p_exp, p_ini_fn_out, p_max_pv, p_min_fold_change, p_cellNo_LB=30, p_no_sep_sampleNo=2):
   #read cell type information for all cells
   L_cid_colNo = 0  # Column index with the cell ID
   L_cell_type_colNo = 1  # Column index with the cell types
   L_sep = '\t'
   L_groups, L_map = read_meta_info_2(p_meta, p_cellNo_LB, L_cid_colNo, L_cell_type_colNo, L_sep)
   
   get_initial_Signature_genes_new_3(p_exp,L_groups,p_ini_fn_out,p_max_pv,p_min_fold_change,p_no_sep_sampleNo)   


def chose_sig_genes_top_std_mean_ratio_3(p_fn, p_topNo, p_fn_out):
   #consider only one cell type when choosing signature genes
   L_rets = {}
   ff = open(p_fn, 'r')
   for ln in ff:
      ln0 = ln.split('\n')
      ln1 = ln0[0].split('\t')
      gene = ln1[0]
      grp = ln1[1]
      t_level = int(ln1[4])
      if not grp in L_rets:
         L_rets[grp] = {}
      if not t_level in L_rets[grp]:
         L_rets[grp][t_level] = []
      L_rets[grp][t_level].append(ln1)
        
   ff.close()
   # for grp in L_rets:
   #    print('---grp#:', grp, len(L_rets[grp]))
   
   L_std_mean_ratio_tp = {}
   for grp in L_rets:
      L_std_mean_ratio_tp[grp] = {}
      for tlv in L_rets[grp]:
         L_std_mean_ratio_tp[grp][tlv] = {}
         for item in L_rets[grp][tlv]:
            t_gene = item[0]
            t_pv = float(item[2])
            t_mean = float(item[5])
            t_std = float(item[6])
            t_missed_ctp = item[7]
            t_std_mean_ratio = t_std/t_mean
            if not t_std_mean_ratio in L_std_mean_ratio_tp[grp][tlv]:
               L_std_mean_ratio_tp[grp][tlv][t_std_mean_ratio] = []
            L_std_mean_ratio_tp[grp][tlv][t_std_mean_ratio].append([t_gene, t_pv, t_missed_ctp, t_mean, t_std])
   

            
   L_gene_top_ratio = {}
   L_gene_top_level = {}
   for grp in L_std_mean_ratio_tp:
       L_gene_top_ratio[grp] = []
       L_gene_top_level[grp] = {}
       t_level_all = []
       for tlv in L_std_mean_ratio_tp[grp]:
          t_level_all.append(tlv)
       t_level_all.sort()
       count = 0
       for tlv in t_level_all:
          t_ratio_all = []
          for t_ratio in L_std_mean_ratio_tp[grp][tlv]:
            t_ratio_all.append(t_ratio)
          t_ratio_all.sort()
          if count<=p_topNo:
             L_gene_top_level[grp][tlv] = 0
             for t_ratio in t_ratio_all:
               for item in L_std_mean_ratio_tp[grp][tlv][t_ratio]:
                   count += 1
                   if count<=p_topNo:
                      #if len(item[2])>0:
                      L_gene_top_ratio[grp].append([item[0], grp, str(tlv), str(count),  str(item[3]), str(item[4]), item[2]])
                      # else:
                      #    L_gene_top_ratio[grp].append([item[0], grp, str(tlv), str(count),  str(item[3]), str(item[4]), 'NA'])
                      L_gene_top_level[grp][tlv] += 1


   for grp in L_gene_top_level:
      if (p_st_using.upper())[0] == 'Y':
         print(grp,' -------')
         for tlv in L_gene_top_level[grp]:
               print('  ',tlv,L_gene_top_level[grp][tlv])
         

   

   fout = open(p_fn_out, 'w')
   output = 'Sig_Gene\tCell_type\tLevel\tRank\tMean\tStd\tCell_type_unseparated\n'
   fout.write(output)
   G_sig_genes = {}; L_cell_type_chosen = {}
   for grp in L_gene_top_ratio:
      L_cell_type_chosen[grp] = 1
      for item in L_gene_top_ratio[grp]:
         G_sig_genes[item[0]] = 1
         output = '\t'.join(item)
         fout.write(output)
         fout.write('\n')
         if len(item[6])>0:
            print(' ****', item[0], grp,'~',item[6])
   fout.close()


   if (p_st_using.upper())[0] == 'Y':     
      print('--- Total gene#:', len(G_sig_genes))
         
   return G_sig_genes, L_cell_type_chosen


def chose_sig_genes_top_std_mean_ratio_3_bak(p_fn, p_topNo = 100):
   #consider only one cell type when choosing signature genes
   L_rets = {}
   ff = open(p_fn, 'r')
   for ln in ff:
      ln0 = ln.split('\n')
      ln1 = ln0[0].split('\t')
      gene = ln1[0]
      grp = ln1[1]
      t_level = int(ln1[4])
      if not grp in L_rets:
         L_rets[grp] = {}
      if not t_level in L_rets[grp]:
         L_rets[grp][t_level] = []
      L_rets[grp][t_level].append(ln1)
        
   ff.close()
   # for grp in L_rets:
   #    print('---grp#:', grp, len(L_rets[grp]))
   
   L_std_mean_ratio_tp = {}
   for grp in L_rets:
      L_std_mean_ratio_tp[grp] = {}
      for tlv in L_rets[grp]:
         L_std_mean_ratio_tp[grp][tlv] = {}
         for item in L_rets[grp][tlv]:
            t_gene = item[0]
            t_pv = float(item[2])
            t_mean = float(item[5])
            t_std = float(item[6])
            t_missed_ctp = item[7]
            t_std_mean_ratio = t_std/t_mean
            if not t_std_mean_ratio in L_std_mean_ratio_tp[grp][tlv]:
               L_std_mean_ratio_tp[grp][tlv][t_std_mean_ratio] = []
            L_std_mean_ratio_tp[grp][tlv][t_std_mean_ratio].append([t_gene,t_pv, t_missed_ctp])
   

            
   L_gene_top_ratio = {}
   L_gene_top_level = {}
   for grp in L_std_mean_ratio_tp:
       L_gene_top_ratio[grp] = []
       L_gene_top_level[grp] = {}
       t_level_all = []
       for tlv in L_std_mean_ratio_tp[grp]:
          t_level_all.append(tlv)
       t_level_all.sort()
       count = 0
       for tlv in t_level_all:
          t_ratio_all = []
          for t_ratio in L_std_mean_ratio_tp[grp][tlv]:
            t_ratio_all.append(t_ratio)
          t_ratio_all.sort()
          if count<=p_topNo:
             L_gene_top_level[grp][tlv] = 0
             for t_ratio in t_ratio_all:
               for item in L_std_mean_ratio_tp[grp][tlv][t_ratio]:
                   count += 1
                   if count<=p_topNo:
                         L_gene_top_ratio[grp].append(item[0])
                         L_gene_top_level[grp][tlv] += 1


   for grp in L_gene_top_level:
      if (p_st_using.upper())[0] == 'Y':
         print(grp,' -------')
         for tlv in L_gene_top_level[grp]:
               print('  ',tlv,L_gene_top_level[grp][tlv])
         

   

   
   G_sig_genes = {}; L_cell_type_chosen = {}
   for grp in L_gene_top_ratio:
      L_cell_type_chosen[grp] = 1
      for gene in L_gene_top_ratio[grp]:
         G_sig_genes[gene] = 1


   if (p_st_using.upper())[0] == 'Y':     
      print('--- Total gene#:', len(G_sig_genes))
         
   return G_sig_genes, L_cell_type_chosen



def get_mean_std_sig_gene_cell_type_new2(p_exp, p_groups, p_top_genes, p_out, p_fn_heatmap, p_sp_in = '\t', p_sp_out = '\t'):
   # L_ini_genes = {}
   # fin = open(p_fn_initial_genes,'r')
   # count = 0
   # for ln in fin:
   #    count += 1
   #    if count>1:
   #       ln1 = ln.split('\t')
   #       L_ini_genes[ln1[0]] = 1
   # fin.close()
   
   L_rets = {}; L_grps = {}
   for grp in p_groups:
      L_grps[grp] = []
   ff = open(p_exp, 'r')
   count = 0
   for ln in ff:
      count += 1
      ln1 = ln.split('\n')
      ln2 = ln1[0].split(p_sp_in)
      if count == 1:
         for idx in range(len(ln2)):
            item = ln2[idx]
            for grp in p_groups:
               if item in p_groups[grp]:
                  L_grps[grp].append(idx)
      else:
            gene = ln2[0]
            #if gene in L_ini_genes:
            if gene in p_top_genes:
               t_grp_exp = {}
               L_rets[gene] = {}
               for grp in p_groups:
                  t_grp_exp[grp] = []
                  for idx in L_grps[grp]:
                     t_grp_exp[grp].append(float(ln2[idx]))
               for grp in p_groups:
                  t_exp = t_grp_exp[grp]
                  tmean = numpy.mean(t_exp)
                  tstd = numpy.std(t_exp, axis=0)
                  L_rets[gene][grp] = [str(tmean), str(tstd)]
                  
   ff.close()
   
   cty_all = []
   for gene in L_rets:
      for ctp in L_rets[gene]:
         cty_all.append(ctp)
      break
   cty_all.sort()
   
   GeneList = []
   Cell_type_gene_mean = {}
   
   ff = open(p_out,'w')
   output = ''
   for ctp in cty_all:
      output += p_sp_out + ctp + '_mean' + p_sp_out +ctp+'_std'
   output += '\n'
   ff.write(output)
   for gene in L_rets:
       GeneList.append(gene)

       tlist = [gene]
       for grp in cty_all:
          if not grp in Cell_type_gene_mean:
             Cell_type_gene_mean[grp] = []
          Cell_type_gene_mean[grp].append(float(L_rets[gene][grp][0]))
          
          #print('\t',grp)
          #try:
          tlist.append(L_rets[gene][grp][0])
          tlist.append(L_rets[gene][grp][1])
          # except:
          #    print('***',gene,L_rets[gene])
       output = p_sp_out.join(tlist)
      # print(output)
       ff.write(output+'\n')

   ff.close()
   
   df_corr = pd.DataFrame(Cell_type_gene_mean)  
   df_corr.index = GeneList
   

   svm = sns.clustermap(df_corr, z_score=0, cmap="vlag", center=0)  
   svm.savefig(p_fn_heatmap, dpi=400)
   plt.close()
   plt.show()


def Get_signature_gene_matrix(p_exp, p_meta, p_ini_sig, p_mean_std_out, p_top_ub, p_fn_heatmap, p_fn_out):
   #read cell type information for all cells
   L_cid_colNo = 0  # Column index with the cell ID
   L_cell_type_colNo = 1  # Column index with the cell types
   L_sep = '\t'
   L_groups, L_map = read_meta_info(p_meta, L_cid_colNo, L_cell_type_colNo, L_sep)


   #Choose the top signature genes
   L_topNo = p_top_ub #Number of to signature genes for cell deconvolution
   L_top_sig_genes, L_Cell_types_Chosen = chose_sig_genes_top_std_mean_ratio_3(p_ini_sig, L_topNo, p_fn_out)
   
   #Get cellID for cell types chosen
   L_groups_new = {}
   for ctp in L_Cell_types_Chosen:
      L_groups_new[ctp] = L_groups[ctp]
      #print('---ctp', ctp, len(L_groups[ctp]))
   
   
   #get mean, std using raw count scRNA-seq file
   get_mean_std_sig_gene_cell_type_new2(p_exp,L_groups_new,L_top_sig_genes,p_mean_std_out, p_fn_heatmap)


def Get_signature_gene_matrix_GeneSet(p_exp, p_meta, p_mean_std_out, p_choose_genes, p_ini_sig, p_fn_heatmap):
   #read cell type information for all cells
   L_cid_colNo = 0  # Column index with the cell ID
   L_cell_type_colNo = 1  # Column index with the cell types
   L_sep = '\t'
   L_groups, L_map = read_meta_info(p_meta, L_cid_colNo, L_cell_type_colNo, L_sep)


   #Choose the top signature genes
   #Choose the top signature genes
   L_topNo = 10 #Number of to signature genes for cell deconvolution
   L_top_sig_genes, L_Cell_types_Chosen = chose_sig_genes_top_std_mean_ratio_3(p_ini_sig, L_topNo)
   
   #Get cellID for cell types chosen
   L_groups_new = {}
   for ctp in L_Cell_types_Chosen:
      L_groups_new[ctp] = L_groups[ctp]
      
   #L_topNo = p_top_ub #Number of to signature genes for cell deconvolution
   L_top_sig_genes = {} # chose_sig_genes_top_std_mean_ratio_3(p_ini_sig, L_topNo)
   for gene in p_choose_genes:
      L_top_sig_genes[gene] = 1
   
   
   #get mean, std using raw count scRNA-seq file
   get_mean_std_sig_gene_cell_type_new2(p_exp,L_groups_new,L_top_sig_genes,p_mean_std_out, p_fn_heatmap)
   


#============================================================================================

def read_sig_gene_mean_std(p_file):
   L_rets = {}
   ctype =  []
   ff = open(p_file, 'r')
   count = 0
   for ln in ff:
      ln1 = ln.split('\n')
      ln2 = ln1[0].split('\t')
      count += 1
      if count == 1:
         for idx in range(len(ln2)):
            if idx%2 == 1:
               t_tp = ln2[idx].split('_')
               t_tp2 = t_tp[0:-1]
               ctype.append('_'.join(t_tp2))

         #print(ctype, len(ctype))
      else:
         gene = ln2[0]
         #if gene in p_genes:
         L_rets[gene] = {}
         for idx in range(len(ctype)):
             t_tp = ctype[idx]
             tmean = ln2[2*idx+1]
             tstd = ln2[2*idx+2]
             L_rets[gene][t_tp] = [float(tmean),float(tstd)]
   ff.close()
   # for gene in L_rets:
   #    print(gene, L_rets[gene])
   
   if (p_st_using.upper())[0] == 'Y':
      print('---- Gene# in the signature matris: ', len(L_rets), ' ---Cell type#:', len(ctype))
      print('  --Cell type names:',ctype)
   return L_rets, ctype

   
def read_bulk_RNA_seq_row_for_sample(p_spInfo, p_gene_in_matrix, p_sp = '\t'):
   L_rets = {}; L_mean_std_new = {}
   ff = open(p_spInfo, 'r')
   L_title  = []
   count = 0
   for ln in ff:
      count += 1
      ln1 = ln.split('\n')
      ln2 = ln1[0].split(p_sp)
      if count == 1:
         L_title = ln2
      else:
         sid = ln2[0]
         L_rets[sid] = {}
         for idx in range(1, len(ln2)):
            gene = L_title[idx]
            if gene in p_gene_in_matrix:
               tva = float(ln2[idx])
               L_rets[sid][gene] = tva
               L_mean_std_new[gene] = p_gene_in_matrix[gene]
   ff.close()
   if (p_st_using.upper())[0] == 'Y':
      print('---- Sample#:', len(L_rets))
      print('---- Number of genes in the matrix: ', len(p_gene_in_matrix), ' ---Common gene number: ', len(L_mean_std_new))
   return L_rets, L_mean_std_new

def read_bulk_RNA_seq_column_for_sample(p_spInfo, p_gene_in_matrix, p_sp = '\t'):
   L_rets = {}; L_mean_std_new = {}
   ff = open(p_spInfo, 'r')
   L_title  = []
   count = 0
   for ln in ff:
      count += 1
      ln1 = ln.split('\n')
      ln2 = ln1[0].split(p_sp)
      if count == 1:
         L_title = ln2
         for idx in range(1,len(ln2)):
            L_rets[ln2[idx]] = {}
      else:
         gene = ln2[0]
         if gene in p_gene_in_matrix:
            L_mean_std_new[gene] = p_gene_in_matrix[gene]
            for idx in range(1, len(ln2)):
               sid = L_title[idx]
               tva = float(ln2[idx])
               L_rets[sid][gene] = tva
   ff.close()
   if (p_st_using.upper())[0] == 'Y':
      print('---- Sample#:', len(L_rets))
      print('---- Number of genes in the matrix: ', len(p_gene_in_matrix), ' ---Common gene number: ', len(L_mean_std_new))
   return L_rets, L_mean_std_new

def get_cell_partition_Linear_Regression_Model(p_expression_RNAseq, p_mean_std):
   L_min_comb = {}

   L_grp_order = []
   for gene in p_mean_std:
      for grp in p_mean_std[gene]:
         L_grp_order.append(grp)
      break
   Y_list = []
   X_list = []
   L_gene_list = []
   for gene in p_expression_RNAseq:
      L_gene_list.append(gene)
   for gene in L_gene_list:
       Y_list.append(p_expression_RNAseq[gene])
       t_list = []
       for grp in   L_grp_order:
         t_list.append(p_mean_std[gene][grp][0])
       X_list.append(t_list)
   x_train = numpy.array(X_list, dtype=numpy.float32)
   y_train = numpy.array(Y_list, dtype=numpy.float32)

   x_train_t = x_train.transpose()
   
   
   x_train_mul = numpy.dot(x_train_t,x_train)
   
   x_train_inv = inv(x_train_mul)
   
   x_result_ma = numpy.dot(x_train_inv, x_train_t)
   
   L_para = numpy.dot(x_result_ma, y_train)
   
   # print('LR:',L_para)
   
   
   L_min_comb = {}; L_min_comb2 = {}
   t_total = 0.0
   for idx in range(len(L_grp_order)):
      gid = L_grp_order[idx]
      va = L_para[idx]
      t_total += va
      L_min_comb[gid] = va

   for idx in range(len(L_grp_order)):
      gid = L_grp_order[idx]
      va = L_para[idx]
      L_min_comb2[gid] = va/t_total*100      
   #L_min_comb = {'MS':L_para[0],'PN':L_para[1], 'CL':L_para[2], 'NOR':L_para[3]}
   #L_min_comb = {'MS':round(L_para[0]/10),'PN':round(L_para[1]/10), 'CL':round(L_para[2]/10), 'NOR':round(L_para[3]/10)}
   # print('LR',L_min_comb)
   
   t_m_list = []; t_va_list = []
   for gene in L_gene_list:
      t_mean = 0.0; t_va = 0.0
      for grp in L_min_comb:
         t_mean += L_min_comb[grp]*p_mean_std[gene][grp][0]
         t_va += L_min_comb[grp]*p_mean_std[gene][grp][1]*p_mean_std[gene][grp][1]
      t_m_list.append(t_mean)
      t_va_list.append(t_va)
   np_t_m_list = numpy.array(t_m_list, dtype=numpy.float32)
   np_t_va_list = numpy.array(t_va_list, dtype=numpy.float32)
   
   x_result_ma_square = x_result_ma*x_result_ma
   
   fr_m = numpy.dot(x_result_ma,np_t_m_list)
   fr_va = numpy.dot(x_result_ma_square,np_t_va_list)
   # print('fr_m',fr_m)
   # print('fr_va',fr_va)
   
   L_dist_fr = {}
   for idx in range(len(L_grp_order)):
      gid = L_grp_order[idx]
      va = fr_va[idx] 
      L_dist_fr[gid] = va
   
   # L_dist_fr = {'MS':[round(L_para[0]/10),fr_va[0]],'PN':[round(L_para[1]/10),fr_va[1]], 'CL':[round(L_para[2]/10),fr_va[2]], 'NOR':[round(L_para[3]/10),fr_va[3]]}
  # t_normalization = 1.0
   #L_dist_fr = {'MS':[round(L_para[0]/t_normalization),fr_va[0]/t_normalization],'PN':[round(L_para[1]/t_normalization),fr_va[1]/t_normalization], 'CL':[round(L_para[2]/t_normalization),fr_va[2]/t_normalization], 'NOR':[round(L_para[3]/t_normalization),fr_va[3]/t_normalization]}
   #L_dist_fr = {'MS':[L_para[0]/t_normalization,fr_va[0]/t_normalization],'PN':[L_para[1]/t_normalization,fr_va[1]/t_normalization], 'CL':[L_para[2]/t_normalization,fr_va[2]/t_normalization], 'NOR':[L_para[3]/t_normalization,fr_va[3]/t_normalization]}
  
   # print(L_dist_fr)
   
   return L_min_comb2

def get_cell_partition_Linear_Regression_Model_2(p_expression_RNAseq, p_mean_std, p_std_gene_old):
   #!!! NOT take absolute fraction value when computing variance
   #Only consider positive fractions when computing variance
   L_min_comb = {}

   L_grp_order = []
   for gene in p_mean_std:
      for grp in p_mean_std[gene]:
         L_grp_order.append(grp)
      break
   Y_list = []
   X_list = []
   L_gene_list = []
   for gene in p_expression_RNAseq:
      L_gene_list.append(gene)
   for gene in L_gene_list:
       t_gene_std = p_std_gene_old[gene] + 0.00000001
       Y_list.append(p_expression_RNAseq[gene]/t_gene_std)
       t_list = []
       for grp in   L_grp_order:
         t_list.append(p_mean_std[gene][grp][0]/t_gene_std)
       X_list.append(t_list)
   x_train = numpy.array(X_list, dtype=numpy.float32)
   y_train = numpy.array(Y_list, dtype=numpy.float32)

   x_train_t = x_train.transpose()
   x_train_mul = numpy.dot(x_train_t,x_train)   
   x_train_inv = inv(x_train_mul)   
   x_result_ma = numpy.dot(x_train_inv, x_train_t)   
   L_para = numpy.dot(x_result_ma, y_train)
   
   # print('LR:',L_para)
   
   
   L_min_comb = {}; L_min_comb2 = {}
   t_total = 0.0
   for idx in range(len(L_grp_order)):
      gid = L_grp_order[idx]
      va = L_para[idx]
      t_total += va
      L_min_comb[gid] = va

   for idx in range(len(L_grp_order)):
      gid = L_grp_order[idx]
      va = L_para[idx]
      #L_min_comb2[gid] = va/t_total*100  
      L_min_comb2[gid] = va/t_total 
     
   L_std_gene_new_dic = {}
   for gene in L_gene_list:
      t_va = 0.0
      for grp in L_min_comb:
         if L_min_comb[grp]>0:
            t_va += L_min_comb[grp]*p_mean_std[gene][grp][1]*p_mean_std[gene][grp][1]
      t_std = t_va**0.5
      L_std_gene_new_dic[gene] = t_std
   
   return L_min_comb2,L_min_comb, L_std_gene_new_dic


def get_cell_partition_Linear_Regression_Model_2_new(p_expression_RNAseq, p_mean_std, p_std_gene_old):
   #take absolute fraction value when computing variance (considering cell types with negative fractions)
   L_min_comb = {}

   L_grp_order = []
   for gene in p_mean_std:
      for grp in p_mean_std[gene]:
         L_grp_order.append(grp)
      break
   Y_list = []
   X_list = []
   L_gene_list = []
   for gene in p_expression_RNAseq:
      L_gene_list.append(gene)
   for gene in L_gene_list:
       t_gene_std = p_std_gene_old[gene] + 0.00000001
       Y_list.append(p_expression_RNAseq[gene]/t_gene_std)
       t_list = []
       for grp in   L_grp_order:
         t_list.append(p_mean_std[gene][grp][0]/t_gene_std)
       X_list.append(t_list)
   #print('----Length of Y:', len(Y_list))
   x_train = numpy.array(X_list, dtype=numpy.float32)
   y_train = numpy.array(Y_list, dtype=numpy.float32)

   x_train_t = x_train.transpose()
   x_train_mul = numpy.dot(x_train_t,x_train)   
   x_train_inv = inv(x_train_mul)   
   x_result_ma = numpy.dot(x_train_inv, x_train_t)   
   L_para = numpy.dot(x_result_ma, y_train)
   
   # print('LR:',L_para)
   
   
   L_min_comb = {}; L_min_comb2 = {}
   t_total = 0.0
   for idx in range(len(L_grp_order)):
      gid = L_grp_order[idx]
      va = L_para[idx]
      t_total += va
      L_min_comb[gid] = va

   for idx in range(len(L_grp_order)):
      gid = L_grp_order[idx]
      va = L_para[idx]
      #L_min_comb2[gid] = va/t_total*100  
      L_min_comb2[gid] = va/t_total 
     
   L_std_gene_new_dic = {}
   for gene in L_gene_list:
      t_va = 0.0
      for grp in L_min_comb:
         #if L_min_comb[grp]>0:
         t_va += abs(L_min_comb[grp])*p_mean_std[gene][grp][1]*p_mean_std[gene][grp][1]
      t_std = t_va**0.5
      L_std_gene_new_dic[gene] = t_std
   
   return L_min_comb2,L_min_comb, L_std_gene_new_dic


def get_cell_partition_Linear_Regression_Model_nnls(p_expression_RNAseq, p_mean_std, p_std_gene_old):
   L_min_comb = {}

   L_grp_order = []
   for gene in p_mean_std:
      for grp in p_mean_std[gene]:
         L_grp_order.append(grp)
      break
   Y_list = []
   X_list = []
   L_gene_list = []
   for gene in p_expression_RNAseq:
      L_gene_list.append(gene)
   for gene in L_gene_list:
       t_gene_std = p_std_gene_old[gene] + 0.00000001
       Y_list.append(p_expression_RNAseq[gene]/t_gene_std)
       t_list = []
       for grp in   L_grp_order:
         t_list.append(p_mean_std[gene][grp][0]/t_gene_std)
       X_list.append(t_list)
   x_train = numpy.array(X_list, dtype=numpy.float32)
   y_train = numpy.array(Y_list, dtype=numpy.float32)

   L_para, this_Res = nnls(x_train,y_train)

   
 #   paraTp = ''
 #   for item in L_para:
 #      paraTp += '\t'+str(item)
 #   print('---LR:',paraTp)
   
   
   L_min_comb = {}; L_min_comb2 = {}
   t_total = 0.0
   for idx in range(len(L_grp_order)):
       gid = L_grp_order[idx]
       va = L_para[idx]
       t_total += va
       L_min_comb[gid] = va

   for idx in range(len(L_grp_order)):
       gid = L_grp_order[idx]
       va = L_para[idx]
       #L_min_comb2[gid] = va/t_total*100  
       L_min_comb2[gid] = va/t_total 
     
   L_std_gene_new_dic = {}
   for gene in L_gene_list:
       t_va = 0.0
       for grp in L_min_comb:
         if L_min_comb[grp]>0:
             t_va += L_min_comb[grp]*p_mean_std[gene][grp][1]*p_mean_std[gene][grp][1]
       t_std = t_va**0.5
       L_std_gene_new_dic[gene] = t_std
   
   return L_min_comb2,L_min_comb, L_std_gene_new_dic


def Percentage_Fraction_nor(p_per, p_fra):
   L_percentage = {}; L_fraction = {}
   L_total = 0.0
   for item in p_per:
      item2 = p_per[item]
      if item2>0:
         L_total += item2
   for item in p_per:
      item2 = p_per[item]
      L_percentage[item] = max(0,item2/L_total)

   for item in p_fra:
      item2 = p_fra[item]
      L_fraction[item] = max(0,item2)
   return L_percentage, L_fraction

def MLMD(p_expression_RNAseq, p_mean_std, p_loop = 6):
   #Minimum Likelihood Model for Deconvolution
   #Fractions are non negative number
   L_percentage_all = {}; L_fractions_all = {}
   t_predic_percent_2 = []; t_fraction_2 = []
   for sp in p_expression_RNAseq:
      t_spot_sp = p_expression_RNAseq[sp]
      L_std_gene_old = {}
      for gene in p_mean_std:
          L_std_gene_old[gene] = 1.0
      for tloop in range(p_loop):
         t_predic_percent_2, t_fraction_2, t_std_gene_2 = get_cell_partition_Linear_Regression_Model_2_new(t_spot_sp, p_mean_std, L_std_gene_old)
         L_percentage_all[sp] = t_predic_percent_2
         L_fractions_all[sp] = t_fraction_2
         L_std_gene_old = t_std_gene_2
      L_per2, L_fra2 = Percentage_Fraction_nor( t_predic_percent_2, t_fraction_2)
      L_percentage_all[sp] = L_per2
      L_fractions_all[sp] = L_fra2
      
   return L_percentage_all, L_fractions_all


def LRM_Least_Square_Error(p_expression_RNAseq, p_mean_std):
   #Linear Regression Modle with Least Squre Error
   #Same with CIBERSORT, but not use the SVR algorithm
   L_percentage_all = {}; L_fractions_all = {}
   for sp in p_expression_RNAseq:
      t_spot_sp = p_expression_RNAseq[sp]
      L_std_gene_old = {}
      for gene in p_mean_std:
          L_std_gene_old[gene] = 1.0
      t_predic_percent_1, t_fraction_1, t_std_gene_1 = get_cell_partition_Linear_Regression_Model_2(t_spot_sp, p_mean_std, L_std_gene_old)
      L_percentage_all[sp] = t_predic_percent_1
      L_fractions_all[sp] = t_fraction_1
      
   return L_percentage_all, L_fractions_all



def MLMD_nns(p_expression_RNAseq, p_mean_std, p_loop = 6):
    #MLMD with non negative fraction predictions
    L_percentage_all = {}; L_fractions_all = {}
    #L_percentage_allB = {}; L_fractions_allB = {}
    for sp in p_expression_RNAseq:
      t_spot_sp = p_expression_RNAseq[sp]
      L_std_gene_old = {}
      for gene in p_mean_std:
          L_std_gene_old[gene] = 1.0
      for t_loop in range(p_loop):
          t_predic_percent_2, t_fraction_2, t_std_gene_2 = get_cell_partition_Linear_Regression_Model_nnls(t_spot_sp, p_mean_std, L_std_gene_old)
          L_percentage_all[sp] = t_predic_percent_2
          L_fractions_all[sp] = t_fraction_2
          L_std_gene_old = t_std_gene_2

      
def Save_results(p_results, p_fn_out):
   Cell_type = []
   #L_sample_list = []; L_ctp_sp_per = {}
   
   fout = open(p_fn_out,'w')
   
   count = 0
   for sp in  p_results:
      count += 1
      #L_sample_list.append(sp)
      if count == 1:
         for ctp in p_results[sp]:
            Cell_type.append(ctp)
            #L_ctp_sp_per[ctp] = []
         Cell_type.sort()
         if (p_st_using.upper())[0] == 'Y':
            print('+++ cell types:', Cell_type)
   
         #output  = 'Sample\Cell_type'
         output  = 'sample/cell_type'
         for item in Cell_type:
            output += '\t'+item
         fout.write(output)
         fout.write('\n')
               
      output = sp
      t_prediction = p_results[sp]
      for ctp in Cell_type:
          output += '\t'+str(t_prediction[ctp])
          #L_ctp_sp_per[sp].append(t_prediction[ctp])

      fout.write(output)
      fout.write('\n')

   fout.close()
   #return L_sample_list, L_ctp_sp_per

     
def ReDeconv(p_sig_matrix, p_bulk_exp, p_fn_save):
   #Our new model for cell type deconvolution
   #read mean and std of top signature genes in each cell type
   L_mean_std, L_cell_type = read_sig_gene_mean_std(p_sig_matrix)  
    
   #read expression data for the mixed samples
   L_samples, L_mean_std_new = read_bulk_RNA_seq_column_for_sample(p_bulk_exp, L_mean_std)
   
   #Find cell type percentages and fractions in all 
   L_predicted_percentages, L_predicted_fractions = MLMD(L_samples, L_mean_std_new,6)

   
   #save cell type perventage predictions
   Save_results(L_predicted_percentages,p_fn_save)


   
def LRM_LSE(p_sig_matrix, p_bulk_exp, p_fn_save):
   #Linear Regression Modle with Least Squre Error
   
   #read mean and std of top signature genes in each cell type
   L_mean_std, L_cell_type = read_sig_gene_mean_std(p_sig_matrix)  
    
   #read expression data for the mixed samples
   L_samples, L_mean_std_new = read_bulk_RNA_seq_column_for_sample(p_bulk_exp, L_mean_std)
   
   #Find cell type percentages and fractions in all 
   L_predicted_percentages, L_predicted_fractions = LRM_Least_Square_Error(L_samples, L_mean_std_new)
   
   #save cell type perventage predictions
   Save_results(L_predicted_percentages,p_fn_save)

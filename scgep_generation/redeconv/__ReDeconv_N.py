# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:30:19 2023

@author: Songjian Lu
"""
import numpy, math, shutil
from numpy.linalg import inv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure



p_st_using = 'Y'

def read_meta_info_sample_ctype(p_file, p_cid_colNo = 0, p_sp_colNo=2, p_type_colNo = 1, p_sp = '\t'):
   L_rets = {}; L_ctype = {}
   ff = open(p_file, 'r')
   count = 0
   for ln in ff:
      count += 1
      if count>1:
         ln0 = ln.split('\n')
         ln1 = ln0[0].split(p_sp)
         ctype = ln1[p_type_colNo]
         cid = ln1[p_cid_colNo]
         spID = ln1[p_sp_colNo]

         L_rets[cid] = [spID, ctype]
         if not ctype in L_ctype:
            L_ctype[ctype] = 0
         L_ctype[ctype] += 1

   ff.close()
   
   if (p_st_using.upper())[0] == 'Y':
      print('--- Counts of different cell types ---')
      print('  cell Type name\tcell number')
   for tp in L_ctype:
      print('  ',tp, '\t\t', L_ctype[tp])

         
   return L_rets

def check_scRNA(p_file,p_st = 0):
   #p_st = 0:org count, 1:log count
   L_rets = {}; L_min = 10000.0; L_max = -1000.0
   L_rets2 = {}
   L_colNo = 0; L_rowNo = 0
   L_title = []
   ff = open(p_file,'r')
   LineNo = 0
   for ln in ff:
      LineNo += 1
   ff.close()
   if (p_st_using.upper())[0] == 'Y':
      print('---- Total row#:', LineNo)
   ff = open(p_file,'r')
   count = 0
   for ln in ff:
       count += 1
       ln1 = ln.split('\n')
       ln2 = ln1[0].split('\t')
       if count % 100 == 0:
          if (p_st_using.upper())[0] == 'Y':
             print(' --- Processing row:', count,'/',LineNo)
       if count<3:
          if (p_st_using.upper())[0] == 'Y':
             print(ln2[0:5],ln2[-5:],len(ln2))
         
         #print(ln)
       if count == 1:
         L_colNo = len(ln2)
         L_title = ln2
         for idx in range(1,len(ln2)):
             L_rets[idx] = 0.0
             L_rets2[idx] = 0
       else:
         for idx in range(1,len(ln2)):
             va = float(ln2[idx])
             va2 = va
             if p_st>0:
               va2 = 2**va
             if va>L_max:
               L_max = va
             if va<L_min:
               L_min = va
             L_rets[idx] += va2
             if va>0:
               L_rets2[idx] += 1
   ff.close()
   L_rowNo = count
   if (p_st_using.upper())[0] == 'Y':
      print('Row#:',L_rowNo, ' Col#:',L_colNo)
      print('Min:',L_min, ' Max:',L_max)
   for idx in range(1,10):
      if (p_st_using.upper())[0] == 'Y':
         print(idx, L_rets[idx])
   return L_rets, L_rets2, L_title


def check_counts_in_sample_type(p_exp_count, p_cid, p_cid_sp_tp):
   L_rets = {}; L_ctype_all = {}; L_cell_tsize = {}
      
   for idx in p_exp_count:
      cid = p_cid[idx]
      t_exp_v = p_exp_count[idx]
      L_cell_tsize[cid] = t_exp_v
      
      cell_sample = p_cid_sp_tp[cid][0]
      cell_type = p_cid_sp_tp[cid][1]
      L_ctype_all[cell_type] = 1
      if not cell_sample in L_rets:
         L_rets[cell_sample] = {}
      if not cell_type in L_rets[cell_sample]:
         L_rets[cell_sample][cell_type] = []
      L_rets[cell_sample][cell_type].append(t_exp_v)

   return L_rets, L_ctype_all, L_cell_tsize


def get_sample_cell_type_exp_mean(p_sp_tp_count, p_fn_out, p_fn_out_2,p_ctypeAll, p_cell_count_LB):
   fout = open(p_fn_out,'w')
   L_tp_list = []
   for tp in p_ctypeAll:
      L_tp_list.append(tp)
   L_tp_list.sort()
   L_sp_list = []
   for sp in p_sp_tp_count:
      L_sp_list.append(sp)
   L_sp_list.sort()
   output = 'sample/cell_type\t' +'\t'.join(L_tp_list)+'\n'
   fout.write(output)
   for sp in L_sp_list:
      output = sp
      for tp in L_tp_list:
         if tp in p_sp_tp_count[sp]:
            t_values = p_sp_tp_count[sp][tp]
            if len(t_values)>=p_cell_count_LB:
               t_mean = numpy.mean(t_values)
               output += '\t'+str(t_mean)
            else:
               output += '\tnan'
         else:
            output += '\tnan'
      output += '\n'
      fout.write(output)
   fout.close()

   fout = open(p_fn_out_2,'w')
   output = 'sample/cell_type\t' +'\t'.join(L_tp_list)+'\n'
   fout.write(output)
   for sp in L_sp_list:
      output = sp
      for tp in L_tp_list:
         if tp in p_sp_tp_count[sp]:
            t_values_No = len(p_sp_tp_count[sp][tp])
            output += '\t'+str(t_values_No)
         else:
            output += '\t0'
      output += '\n'
      fout.write(output)  
   fout.close()


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
   
   L_cell_all = {}; L_cell_list = []; L_sample_all = {}
   ff = open(p_meta, 'r')
   count = 0
   for ln in ff:
      count += 1
      if count>1:
         ln1 = ln.split('\t')
         L_cell_all[ln1[0]] = 1
         L_cell_list.append(ln1[0])
         if len(ln1)>2:
             L_sample_all[ln1[2]] = 1            
   ff.close()
   
   if len(L_sample_all)<2:
       L_rets = -1
       output = ''
       if len(L_sample_all)==1:
         output += '\n !!! The scRNA-seq data only has one sample/subject ---'
         output += '\n !!! There is no need for the scRNA-seq data normalization ---'
       else:
         output += '\n !!! The meta data does not have the third column for the sample/subject information ---'
         output += '\n !!! There is not enough information for performing the scRNA-seq data normalization --- '

       output += '\n !!! Please check the input data files before run the program --- '
               
       print(output)
       return L_rets
   
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

def get_sample_cell_type_exp_mean_and_cell_count(p_meta, p_exp, p_out_mean, p_out_count, p_out_cell_size, p_cell_count_LB = 10):
   L_cell_types =  read_meta_info_sample_ctype(p_meta)
   L_cell_read_counts, L_cell_expressed_gene_counts, L_title = check_scRNA(p_exp)
   L_count_cell_sample_types, L_all_types, L_cell_transcriptome_size = check_counts_in_sample_type(L_cell_read_counts,L_title,L_cell_types)
   get_sample_cell_type_exp_mean(L_count_cell_sample_types, p_out_mean, p_out_count, L_all_types, p_cell_count_LB)
   
   fout = open(p_out_cell_size, 'w')
   for cid in L_cell_transcriptome_size:
      output = cid+'\t'+str(L_cell_transcriptome_size[cid])+'\n'
      fout.write(output)
   fout.close()

def read_sp_tp_mean(p_file):
   L_rets = {}; L_gid_idx = {}
   ff = open(p_file,'r')
   count = 0
   for ln in ff:
      count += 1
      if count>1:
         ln1 = ln.split('\n')
         ln2 = ln1[0].split('\t')
         gid = ln2[0]
         L_rets[gid] = []
         L_gid_idx[count-2] = gid
         for idx in  range(1,len(ln2)):
            item = ln2[idx]
            if item == 'nan':
               L_rets[gid].append(-10000)
            else:
               L_rets[gid].append(float(item))
   ff.close()
   # print('----------------------')
   # print('Index\t\tSample_name')
   # for idx in L_gid_idx:
   #    print(' ',idx,'\t\t',L_gid_idx[idx])

   return L_rets


def get_means_two_samples(p_mean_all, p_sp1, p_sp2):
   L_sp_list_all = []
   for sp in p_mean_all:
      L_sp_list_all.append(sp)
   L_sp_list_all.sort()
   

   L_sp1 = L_sp_list_all[p_sp1]
   L_sp2 = L_sp_list_all[p_sp2]

   if (p_st_using.upper())[0] == 'Y':   
      print(' --- Draw grap for samples:', L_sp1, 'and', L_sp2)

   L_data_1 = p_mean_all[L_sp1]
   L_data_2 = p_mean_all[L_sp2]

   Y_list = []
   X_list = []
   for idx in range(len(L_data_1)):
       v1 = L_data_1[idx]
       v2 = L_data_2[idx]
       if v1>=0 and v2>=0:
         X_list.append(v1)
         Y_list.append(v2)
   return X_list, Y_list

def get_LR(p_sp_tp_mean, p_chosen_s, p_fix_idx = 0):
   L_rets = {}
   L_fixed_sp = ''
   count = -1
   L_sp_list = []
   for sp in p_sp_tp_mean:
      L_sp_list.append(sp)
   L_sp_list.sort()
   for sp in L_sp_list:
      count += 1
      if count == p_fix_idx:
         L_fixed_sp = sp
         break
   if (p_st_using.upper())[0] == 'Y':
      print('---Fixed sp:', L_fixed_sp)
   
   L_data_1 = p_sp_tp_mean[L_fixed_sp]
   for sp in p_chosen_s:
      L_data_2 = p_sp_tp_mean[sp]
      

      Y_list = []
      X_list = []
      X_tp = []
      for idx in range(len(L_data_1)):
          v1 = L_data_1[idx]
          v2 = L_data_2[idx]
          if v1>0 and v2>0:
            X_list.append([v1,1])
            Y_list.append(v2)
            X_tp.append(v1)
            
      # print('\n---+++')
      # print(L_fixed_sp, (X_tp))
      # print(sp, (Y_list))      
      
      x_train = numpy.array(X_list, dtype=numpy.float32)
      y_train = numpy.array(Y_list, dtype=numpy.float32)
   
      x_train_t = x_train.transpose()
      x_train_mul = numpy.dot(x_train_t,x_train)   
      x_train_inv = inv(x_train_mul)   
      x_result_ma = numpy.dot(x_train_inv, x_train_t)   
      L_para = numpy.dot(x_result_ma, y_train)
      
      L_rets[sp] = [L_para[0],L_para[1]]
      # print('a=',L_para[0], 'b=', L_para[1])
   return L_rets

def scRNA_seq_normalization(p_exp, p_exp_out, p_meta, p_sp_ctp_tra_size_mean, p_cell_size, p_chosen_samples, p_fixed_sample = 0):
   L_sp_tp_mean = read_sp_tp_mean(p_sp_ctp_tra_size_mean)
   L_Para_LR = get_LR(L_sp_tp_mean, p_chosen_samples, p_fixed_sample)
    
   L_v_max = -100; L_a_min = 100
   count = 0
   for sp in L_Para_LR:
     # if (p_st_using.upper())[0] == 'Y':
     #    print(sp,'\ta =',L_Para_LR[sp][0],'\tb =',L_Para_LR[sp][1],'\tratio =',L_Para_LR[sp][1]/L_Para_LR[sp][0])
     if L_Para_LR[sp][0] < L_a_min:
         L_a_min = L_Para_LR[sp][0]
     count += 1
     if count == 1:
         L_v_max = L_Para_LR[sp][1]/L_Para_LR[sp][0]
     else:
         t_ratio = L_Para_LR[sp][1]/L_Para_LR[sp][0]
         if L_v_max<t_ratio:
           L_v_max = t_ratio
     #print('--max ratio: ',L_v_max)
   if L_a_min<=0:
     if (p_st_using.upper())[0] == 'Y':

         print('!!! As at least one of constant "a" is negative, transcriptome sizes of cell types in at least two samples have problems.\n')
         print('    It is likely the cell type classification is not purified enough. The normalization will not perform.')
     return 0
   
   ff = open(p_exp,'r')
   LineNo = 0
   for ln in ff:
     LineNo += 1
   ff.close()
   
   LineNo2 = LineNo
   #print('---- Total row#:', LineNo)  
   
   L_Para_LR_2 = {}
   for sp in L_Para_LR:
     L_Para_LR_2[sp] = [L_Para_LR[sp][0],L_Para_LR[sp][1],(L_v_max-L_Para_LR[sp][1]/L_Para_LR[sp][0])/LineNo2]
     #print('  ', sp, [L_Para_LR[sp][0],L_Para_LR[sp][1],(L_v_max-L_Para_LR[sp][1]/L_Para_LR[sp][0])/LineNo])
    
   L_cell_types =  read_meta_info_sample_ctype(p_meta)
   count = 0
   for cid in L_cell_types:
     count += 1
     # if count<20:
     #    if (p_st_using.upper())[0] == 'Y':
     #       print(cid, L_cell_types[cid], L_Para_LR_2[ L_cell_types[cid][0]])
       
   fout = open(p_exp_out, 'w')
   ff = open(p_exp,'r')
   count = 0
   L_title = []
   for ln in ff:
     count += 1
     ln1 = ln.split('\n')
     ln2 = ln1[0].split('\t')
     if count%100 == 1:
        if (p_st_using.upper())[0] == 'Y':
           print('--- Processing row:',count,'/',LineNo, 'for normlaization......')
     if count == 1:
       L_title = ln2
       fout.write(ln)
     else:
       t_list = [ln2[0]]
       for idx in range(1, len(ln2)):
           t_cid = L_title[idx]
           
           t_cell_size = p_cell_size[t_cid]
           
           t_spid = L_cell_types[t_cid][0]
           t_v_a = L_Para_LR_2[t_spid][0]
           t_shift = L_Para_LR_2[t_spid][2]
           t_v_org = float(ln2[idx])
           
           #Add shift to all genes
           t_v_new = t_v_org/t_v_a + t_shift
           t_list.append(str(t_v_new))
           
           #Add shift to genes with positive expression
           # if t_cell_size>0:
           #    t_v_new = t_v_org/t_v_a*(1 + t_shift*t_v_a/t_cell_size)              
           #    t_list.append(str(t_v_new))
           # else:
           #    t_list.append(str(t_v_org))
       output = '\t'.join(t_list)+'\n'
       fout.write(output)
   ff.close()
   fout.close()
         

def get_LR_All(p_sp_tp_mean, p_fix_idx = 0):
   L_rets = {}
   L_fixed_sp = ''
   count = -1
   L_sp_list = []
   for sp in p_sp_tp_mean:
      L_sp_list.append(sp)
   L_sp_list.sort()
   for sp in L_sp_list:
      count += 1
      if count == p_fix_idx:
         L_fixed_sp = sp
         break
   if (p_st_using.upper())[0] == 'Y':
      print('---Fixed sp:', L_fixed_sp)
   
   L_data_1 = p_sp_tp_mean[L_fixed_sp]
   for sp in p_sp_tp_mean:
      L_data_2 = p_sp_tp_mean[sp]
      

      Y_list = []
      X_list = []
      X_tp = []
      for idx in range(len(L_data_1)):
          v1 = L_data_1[idx]
          v2 = L_data_2[idx]
          if v1>0 and v2>0:
            X_list.append([v1,1])
            Y_list.append(v2)
            X_tp.append(v1)
            
      # print('\n---+++')
      # print(L_fixed_sp, (X_tp))
      # print(sp, (Y_list))      
      
      x_train = numpy.array(X_list, dtype=numpy.float32)
      y_train = numpy.array(Y_list, dtype=numpy.float32)
   
      x_train_t = x_train.transpose()
      x_train_mul = numpy.dot(x_train_t,x_train)   
      x_train_inv = inv(x_train_mul)   
      x_result_ma = numpy.dot(x_train_inv, x_train_t)   
      L_para = numpy.dot(x_result_ma, y_train)
      
      L_rets[sp] = [L_para[0],L_para[1]]
      # print('a=',L_para[0], 'b=', L_para[1])
   return L_rets

def scRNA_seq_normalization_All(p_exp, p_exp_out, p_meta, p_sp_ctp_tra_size_mean, p_cell_size, p_fixed_sample = 0):
   L_sp_tp_mean = read_sp_tp_mean(p_sp_ctp_tra_size_mean)
   L_Para_LR = get_LR_All(L_sp_tp_mean, p_fixed_sample)
    
   L_v_max = -100; L_a_min = 100
   count = 0
   for sp in L_Para_LR:
     # if (p_st_using.upper())[0] == 'Y':
     #    print(sp,'\ta =',L_Para_LR[sp][0],'\tb =',L_Para_LR[sp][1],'\tratio =',L_Para_LR[sp][1]/L_Para_LR[sp][0])
     if L_Para_LR[sp][0] < L_a_min:
         L_a_min = L_Para_LR[sp][0]
     count += 1
     if count == 1:
         L_v_max = L_Para_LR[sp][1]/L_Para_LR[sp][0]
     else:
         t_ratio = L_Para_LR[sp][1]/L_Para_LR[sp][0]
         if L_v_max<t_ratio:
           L_v_max = t_ratio
     #print('--max ratio: ',L_v_max)
   if L_a_min<=0:
     if (p_st_using.upper())[0] == 'Y':

         print('!!! As at least one of constant "a" is negative, transcriptome sizes of cell types in at least two samples have problems.\n')
         print('    It is likely the cell type classification is not purified enough. The normalization will not perform.')
     return 0
   
   ff = open(p_exp,'r')
   LineNo = 0
   for ln in ff:
     LineNo += 1
   ff.close()
   LineNo2 = LineNo
   #print('---- Total row#:', LineNo)  
   
   L_Para_LR_2 = {}
   for sp in L_Para_LR:
     L_Para_LR_2[sp] = [L_Para_LR[sp][0],L_Para_LR[sp][1],(L_v_max-L_Para_LR[sp][1]/L_Para_LR[sp][0])/LineNo2]
     #print('  ', sp, [L_Para_LR[sp][0],L_Para_LR[sp][1],(L_v_max-L_Para_LR[sp][1]/L_Para_LR[sp][0])/LineNo])
    
   L_cell_types =  read_meta_info_sample_ctype(p_meta)
   count = 0
   for cid in L_cell_types:
     count += 1
     # if count<20:
     #    if (p_st_using.upper())[0] == 'Y':
     #       print(cid, L_cell_types[cid], L_Para_LR_2[ L_cell_types[cid][0]])
       
   fout = open(p_exp_out, 'w')
   ff = open(p_exp,'r')
   count = 0
   L_title = []
   for ln in ff:
     count += 1
     ln1 = ln.split('\n')
     ln2 = ln1[0].split('\t')
     if count%100 == 1:
        if (p_st_using.upper())[0] == 'Y':
           print('--- Processing row:',count,'/',LineNo, 'for normlaization......')
     if count == 1:
       L_title = ln2
       fout.write(ln)
     else:
       t_list = [ln2[0]]
       for idx in range(1, len(ln2)):
           t_cid = L_title[idx]
           
           t_cell_size = p_cell_size[t_cid]
           
           t_spid = L_cell_types[t_cid][0]
           t_v_a = L_Para_LR_2[t_spid][0]
           t_shift = L_Para_LR_2[t_spid][2]
           t_v_org = float(ln2[idx])

           #Add shift to all genes
           t_v_new = t_v_org/t_v_a + t_shift
           t_list.append(str(t_v_new))
           
           #Add shift to genes with positive expression
           # if t_cell_size>0:
           #    t_v_new = t_v_org/t_v_a*(1 + t_shift*t_v_a/t_cell_size)
           #    t_list.append(str(t_v_new))
           # else:
           #    t_list.append(str(t_v_org))
           
       output = '\t'.join(t_list)+'\n'
       fout.write(output)
   ff.close()
   fout.close()
      
def get_means_two_samples_2(p_mean_all, p_sp1, p_sp2):

   L_data_1 = p_mean_all[p_sp1]
   L_data_2 = p_mean_all[p_sp2]

   Y_list = []
   X_list = []
   for idx in range(len(L_data_1)):
       v1 = L_data_1[idx]
       v2 = L_data_2[idx]
       if v1>0 and v2>0:
         X_list.append(v1)
         Y_list.append(v2)
   return X_list, Y_list

def get_Pearson_correlation_samples(p_sp_tp_mean, p_cor_lb=0.95):
   L_Pearson = {}
   L_sp_list = []
   L_Pearson_spNo = {}
   L_cor_lb = 0.95
   for sp in p_sp_tp_mean:
       L_sp_list.append(sp)
   L_sp_list.sort()
   for tsp1 in L_sp_list:
       L_Pearson[tsp1] = []
       L_Pearson_spNo[tsp1] = []
       for tsp2 in L_sp_list:
          L_v1, L_v2 =  get_means_two_samples_2(p_sp_tp_mean, tsp1, tsp2)
          if len(L_v1)>3:
            t_pc = numpy.corrcoef(L_v1, L_v2)[0][1]
            L_Pearson[tsp1].append(t_pc)
            if t_pc>=p_cor_lb:
               L_Pearson_spNo[tsp1].append(tsp2)
          else:
            L_Pearson[tsp1].append(0)
   # print('----------------------')
   # print('Index\t\tSample_name\tStrong_correlation_Sample#')            
   # for idx in range(len(L_sp_list)):
   #    tsp = L_sp_list[idx]
   #    print(idx,'\t',tsp,'\t',len(L_Pearson_spNo[tsp]))
   #    print('   ---', L_Pearson_spNo[tsp])
   return L_Pearson, L_Pearson_spNo


def draw_heatmap(p_Pearson):
   df_corr = pd.DataFrame(p_Pearson)
   #df.columns = L_sp_list
   L_col_name = list(df_corr.columns)
   df_corr.index = L_col_name
   
   
   sns.light_palette("blue", as_cmap=True)
   #sns.color_palette("light:b", as_cmap=True)
   ax = sns.heatmap(df_corr, annot=False)
   
   return ax, df_corr


def draw_heatmap_Pearson_all(p_fn_ctyp_mean, p_fn_heatmap, p_fn_matrix):
   L_sp_tp_mean = read_sp_tp_mean(p_fn_ctyp_mean)
   L_Pearson, L_samples = get_Pearson_correlation_samples(L_sp_tp_mean, 0.95)
   #print(' --- Total sample#:', len(L_sp_tp_mean))

   L_figure, L_matrix = draw_heatmap(L_Pearson)
   L_heatmap = L_figure.get_figure()
   L_heatmap.savefig(p_fn_heatmap, dpi=400)
   #plt.savefig(p_fn_heatmap, dpi=400)
   
   L_matrix.to_csv(p_fn_matrix)
   
   
def draw_point(p_sp_tp_mean, p_idx1, p_idx2):
 
   L_X, L_Y = get_means_two_samples(p_sp_tp_mean, p_idx1, p_idx2)

   #draw the expression means of all cell types in two chosen samples 
   
   print('X:', L_X)
   print('Y:', L_Y)
   plt.plot(L_X, L_Y, 'o')
   plt.show()
   return L_X, L_Y

def draw_cell_type_size_mean_two_samples(p_fn_ctyp_mean):
   L_sp_tp_mean = read_sp_tp_mean(p_fn_ctyp_mean)
   
   
   #Choose any two samples to draw point graph
   t_sp_1_idx = int(input("Input the index of sample1 (0-"+str(len(L_sp_tp_mean)-1)+"): "))
   t_sp_2_idx = int(input("Input the index of sample2 (0-"+str(len(L_sp_tp_mean)-1)+"): "))
   tX, tY = draw_point(L_sp_tp_mean, t_sp_1_idx, t_sp_2_idx)

def get_sample_baseline(p_fn_ctyp_mean, L_Pearson_LB):
   L_sp_tp_mean = read_sp_tp_mean(p_fn_ctyp_mean)
   L_Pearson, L_samples = get_Pearson_correlation_samples(L_sp_tp_mean, L_Pearson_LB)
   L_sp_list = []
   for sp in L_samples:
      L_sp_list.append(sp)
   L_sp_list.sort()
   
   L_rets = 0
   L_lenTp = len(L_samples[L_sp_list[0]])
   for idx in range(len(L_sp_list)):
      tsp = L_sp_list[idx]
      if L_lenTp<len(L_samples[tsp]):
         L_rets = idx
         L_lenTp = len(L_samples[tsp])
   return L_rets

def draw_cell_type_size_mean_point_plot(p_fn_ctyp_mean, p_fn_point, L_base, L_Pearson_LB, p_gn_row=3):
   L_sp_tp_mean = read_sp_tp_mean(p_fn_ctyp_mean)
   L_Pearson, L_samples = get_Pearson_correlation_samples(L_sp_tp_mean, L_Pearson_LB)
   L_sp_list = []
   for sp in L_samples:
      L_sp_list.append(sp)
   L_sp_list.sort()
   L_sp_base = L_sp_list[L_base]
   L_chosen_sp = L_samples[L_sp_base]
   #print(L_sp_base, L_chosen_sp)
   
   if len(L_chosen_sp)<2:
      print('\n---- Only one sample is chosen, no point plot figure!!!! \n')
      return 1
   
   count = 0
   colNo = p_gn_row
   rowNo = math.ceil((len(L_chosen_sp)-1)/colNo)
   
   if len(L_chosen_sp)>2:
   
      fig, axs = plt.subplots(rowNo, colNo, figsize=(6*colNo,4.5*rowNo))
      for t_sp2 in L_chosen_sp:
         if not t_sp2 == L_sp_base:
            L_v1, L_v2 =  get_means_two_samples_2(L_sp_tp_mean, L_sp_base, t_sp2)
            t_c_idx = count % colNo
            t_r_idx = math.floor(count/colNo)
            # print('\n-----',L_sp_base,t_sp2, t_r_idx, t_c_idx)
            # print(L_v1)
            # print(L_v2)
            #axs[t_r_idx,t_c_idx].plot(L_v1, L_v2, 'o')
            if colNo == 1:
               axs[t_r_idx].plot(L_v1, L_v2, 'o')
               axs[t_r_idx].set_xlabel(L_sp_base)
               axs[t_r_idx].set_ylabel(t_sp2)
            else:
               if rowNo == 1:
                  axs[t_c_idx].plot(L_v1, L_v2, 'o')
                  axs[t_c_idx].set_xlabel(L_sp_base)
                  axs[t_c_idx].set_ylabel(t_sp2)
               else:
                  axs[t_r_idx,t_c_idx].plot(L_v1, L_v2, 'o')
                  axs[t_r_idx,t_c_idx].set_xlabel(L_sp_base)
                  axs[t_r_idx,t_c_idx].set_ylabel(t_sp2)
            count += 1
   
            
      fig.savefig(p_fn_point, dpi=400)
   else:
      if len(L_chosen_sp) == 2:
            for t_sp2 in L_chosen_sp:
               if not t_sp2 == L_sp_base:
                  L_v1, L_v2 =  get_means_two_samples_2(L_sp_tp_mean, L_sp_base, t_sp2)

                  
                            
                  figure = Figure(figsize=(8,7.5), dpi=100)
                  a = figure.add_subplot(111)
                  a.plot(L_v1, L_v2, 'o')    
                  figure.savefig(p_fn_point, dpi=400)
                  
                  plt.plot(L_v1, L_v2, 'o') 
      
def get_sp_ctp_cellNo(p_sp_ctp, p_samples):
   L_rets = {}
   for sp in p_samples:
      for ctp in p_sp_ctp[sp]:
         if not ctp in L_rets:
            L_rets[ctp] = 0
         L_rets[ctp] += p_sp_ctp[sp][ctp]
   L_ctp_list = []
   for ctp in L_rets:
      L_ctp_list.append(ctp)
   L_ctp_list.sort()
   output1 = ''
   output2 = ''
   for ctp in L_ctp_list:
      output1 += '\t'+ctp
      output2 += '\t'+str(L_rets[ctp])
   return output1, output2
   
   
   
def get_cellType_cellNumber(p_meta, p_samples_Pearson):
   L_sample_cellType_map = {}
   ff = open(p_meta, 'r')
   count = 0
   for ln in ff:
      count += 1
      if count>1:
         ln1 = ln.split('\n')
         ln2 = ln1[0].split('\t')
         tctp = ln2[1]
         tspid = ln2[2]
         if not tspid in L_sample_cellType_map:
            L_sample_cellType_map[tspid] = {}
         if not tctp in L_sample_cellType_map[tspid]:
            L_sample_cellType_map[tspid][tctp] = 0
         L_sample_cellType_map[tspid][tctp] += 1
   ff.close()
   L_sp_list = []
   for sp in p_samples_Pearson:
      L_sp_list.append(sp)
   L_sp_list.sort()
   
   
   for idx in range(len(L_sp_list)):
      tsp = L_sp_list[idx]
      print('\n',idx,'\t',tsp,'\t',len(p_samples_Pearson[tsp]),p_samples_Pearson[tsp])
      t_output1, t_output2 = get_sp_ctp_cellNo(L_sample_cellType_map, p_samples_Pearson[tsp])
      print(t_output1)
      print(t_output2 )
      #print(p_samples_Pearson[tsp])


def get_sample_cell_type_information_top_Pearson(p_fn_meta, p_fn_ctyp_mean, p_Pearson_LB = 0.7):
   L_sp_tp_mean = read_sp_tp_mean(p_fn_ctyp_mean)
   L_Pearson, L_samples = get_Pearson_correlation_samples(L_sp_tp_mean, p_Pearson_LB)
   # print(' --- Total sample#:', len(L_sp_tp_mean))

   
   get_cellType_cellNumber(p_fn_meta, L_samples)   

def get_cellType_cellNumber_2(p_count, p_samples_Pearson):
   L_sample_cellType_map = {}
   
   L_rets = {}
   L_title = []
   ff = open(p_count, 'r')
   count = 0
   for ln in ff:
      ln1 = ln.split('\n')
      ln2 = ln1[0].split('\t')
      count += 1
      if count == 1:
         L_title = ln2
      else:
         tsp = ln2[0]
         L_sample_cellType_map[tsp] = {}
         for idx in range(1, len(ln2)):
            tpid = L_title[idx]
            t_cno = int(ln2[idx])
            L_sample_cellType_map[tsp][tpid] = t_cno
   ff.close()  
   
   L_sp_list = []
   for sp in p_samples_Pearson:
      L_sp_list.append(sp)
   L_sp_list.sort()
   
   output_All = ''
   for idx in range(len(L_sp_list)):
      tsp = L_sp_list[idx]
      #print('\n',idx,'\t',tsp,'\t',len(p_samples_Pearson[tsp]),p_samples_Pearson[tsp])
      output = '\n'+str(idx) +'\t' + tsp +'\t' + str(len(p_samples_Pearson[tsp])) + '\t'+ str(p_samples_Pearson[tsp]) +'\n'
      output_All += output
      
      
      t_output1, t_output2 = get_sp_ctp_cellNo(L_sample_cellType_map, p_samples_Pearson[tsp])
      #print(t_output1)
      output = t_output1+'\n'
      output_All += output
      #print(t_output2)
      output = t_output2+'\n'
      output_All += output
      #print(p_samples_Pearson[tsp])
   return output_All

def get_sample_cell_type_information_top_Pearson_2(p_fn_count, p_fn_ctyp_mean, p_fn_extra_info, p_Pearson_LB = 0.7):
   L_sp_tp_mean = read_sp_tp_mean(p_fn_ctyp_mean)
   L_Pearson, L_samples = get_Pearson_correlation_samples(L_sp_tp_mean, p_Pearson_LB)
   # print(' --- Total sample#:', len(L_sp_tp_mean))

   
   L_output_All = get_cellType_cellNumber_2(p_fn_count, L_samples)
   
   fout = open(p_fn_extra_info, 'w')
   fout.write(L_output_All)
   fout.close()  
   print(L_output_All)

   
def get_chosen_samples_heatmap(p_base, p_samples_Pearson, p_threshold, p_sp_ctp_mean):
   L_sp_list = []
   L_rets = []; L_sp_ctp_mean2 = {}
   for sp in p_samples_Pearson:
      L_sp_list.append(sp)
   L_sp_list.sort()
   L_sp_base = L_sp_list[p_base]
   for idx in range(len(p_samples_Pearson[L_sp_base])):
      tva = p_samples_Pearson[L_sp_base][idx]
      if tva>=p_threshold:
         L_rets.append(L_sp_list[idx])
         
   for sp in L_rets:
      L_sp_ctp_mean2[sp] = p_sp_ctp_mean[sp]
   print('---Chose samples:', L_rets, len(L_rets))
   #print('***', p_sp_ctp_mean)
      
   L_Pearson2, L_samples2 = get_Pearson_correlation_samples(L_sp_ctp_mean2, p_threshold)
   draw_heatmap(L_Pearson2)
   
   return L_rets

def Make_small_meta_data_sample(p_fn_meta_small,p_fn_meta_small_malig, p_sp_chosen):
   fout = open(p_fn_meta_small_malig, 'w')
   output = 'Cell_ID\tCell_type\tSample_ID\n'
   fout.write(output)
   ff = open(p_fn_meta_small,'r')
   count = 0
   for ln in ff:
      ln1 = ln.split('\n')
      ln2 = ln1[0].split('\t')
      ctp = ln2[2]
      if ctp in p_sp_chosen:
         fout.write(ln)
         count += 1

   ff.close()
   fout.close()
   print('--Total chosen cell#:', count)
      
def get_sub_matrix_sp_col_2(p_fn_exp, p_fn_meta, p_fn_out):
   #The order of columnn in the submatrix is the order of row in mata data
   L_sp_chosen = {}; L_sp_order = []
   L_cellID_idx_map = {}
   ff = open(p_fn_meta,'r')
   count = 0
   for ln in ff:
      count += 1
      if count>1:
         ln1 = ln.split('\t')
         L_sp_chosen[ln1[0]] = 1
         L_sp_order.append(ln1[0])
   ff.close()
   #print('--- Total cell# info read:', count-1)
   fout = open(p_fn_out,'w')
   ff = open(p_fn_exp,'r')
   count = 0; count2 = 0
   #L_idx_in = []
   for ln in ff:
      count += 1
      ln1 = ln.split('\n')
      ln2 = ln1[0].split('\t')
      if count%100 == 1:
         print('---Procesing row', count, ' for new matrix ......')
      if count == 1:
         output = 'Gene_sample'
         for idx in range(1,len(ln2)):
            tcid = ln2[idx]
            L_cellID_idx_map[tcid] = idx
         for tcid in L_sp_order:
            output += '\t'+tcid
            #L_idx_in.append(idx)
         output += '\n'
         fout.write(output)
      else:
         output = ln2[0]
         for tcid in L_sp_order:
            tidx = L_cellID_idx_map[tcid]
            output += '\t'+ln2[tidx]
         output += '\n'
         fout.write(output)
            
         

            
   ff.close()
   fout.close()
   #print('---- total cell#:', len(L_idx_in))   

def get_meta_and_exp_chosen_sample(p_exp, p_meta, p_exp_out, p_meta_out, p_chosen_sp):
   L_sp_dic = {}
   for sp in p_chosen_sp:
      L_sp_dic[sp] = 1
   Make_small_meta_data_sample(p_meta, p_meta_out, L_sp_dic)
   
   print('--- Making the new scRNA-seq data matrix for chosen samples ---')
   get_sub_matrix_sp_col_2(p_exp, p_meta_out, p_exp_out)  
   
      
def get_and_normalize_cell_subset(p_fn_exp, p_fn_meta, p_fn_ctyp_mean, p_fn_cell_size, p_fn_exp_2, p_fn_meta_2, p_fn_exp_3, p_L_sample_based, p_L_Pearson_LB):
   L_sp_tp_mean = read_sp_tp_mean(p_fn_ctyp_mean)
   L_Pearson, L_samples = get_Pearson_correlation_samples(L_sp_tp_mean, p_L_Pearson_LB)


   L_sample1 = get_chosen_samples_heatmap(p_L_sample_based, L_Pearson, p_L_Pearson_LB, L_sp_tp_mean)

   print('--- Get cell subset......')
   get_meta_and_exp_chosen_sample(p_fn_exp, p_fn_meta, p_fn_exp_2, p_fn_meta_2, L_sample1)
   
   print('--- Normalization.....')
   L_cell_size = read_cell_transcriptome_size(p_fn_cell_size)
   scRNA_seq_normalization(p_fn_exp_2, p_fn_exp_3, p_fn_meta_2, p_fn_ctyp_mean, L_cell_size, L_sample1,  p_L_sample_based)

def read_cell_transcriptome_size(p_file):
   L_rets = {}
   ff = open(p_file, 'r')
   for ln in ff:
      ln1 = ln.split('\n')
      ln2 = ln1[0].split('\t')
      cid = ln2[0]
      tsize = float(ln2[1])
      L_rets[cid] = tsize
   ff.close()
   return L_rets

def get_cell_subset_scRNA_seq_data_normalization(p_fn_exp, p_fn_meta, p_fn_ctyp_mean, p_fn_cell_size, p_fn_exp_2, p_fn_meta_2, p_fn_exp_3, p_L_sample_based, p_L_Pearson_LB):
   L_sp_tp_mean = read_sp_tp_mean(p_fn_ctyp_mean)
   L_Pearson, L_samples = get_Pearson_correlation_samples(L_sp_tp_mean, p_L_Pearson_LB)


   L_sample1 = get_chosen_samples_heatmap(p_L_sample_based, L_Pearson, p_L_Pearson_LB, L_sp_tp_mean)
   
   print(len(L_sp_tp_mean), len(L_samples), len(L_sample1))
   
   L_cell_size = read_cell_transcriptome_size(p_fn_cell_size)
   
   if len(L_sample1) == len(L_sp_tp_mean):
      shutil.copyfile(p_fn_meta, p_fn_meta_2)
      if len(L_sample1)>1:
         print('--- Do normalization.....')
         scRNA_seq_normalization_All(p_fn_exp, p_fn_exp_3, p_fn_meta, p_fn_ctyp_mean, L_cell_size,  p_L_sample_based)
      else:
         output = '\n!!!! Only one sample is chosen. No need for doing normalization ------'
         shutil.copyfile(p_fn_exp, p_fn_exp_3)
         print(output)
   else:
      
      print('--- Get cell subset......')
      get_meta_and_exp_chosen_sample(p_fn_exp, p_fn_meta, p_fn_exp_2, p_fn_meta_2, L_sample1)
      
      if len(L_sample1)>1:
         print('--- Do normalization.....')
         scRNA_seq_normalization(p_fn_exp_2, p_fn_exp_3, p_fn_meta_2, p_fn_ctyp_mean, L_cell_size, L_sample1,  p_L_sample_based)
      else:
         output = '\n!!!! Only one sample is chosen. No need for doing normalization ------'
         shutil.copyfile(p_fn_exp_2, p_fn_exp_3)
         print(output)

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:27:43 2023

@author: Songjian Lu
"""

import time


from __ReDeconv_P import *


     
print('********************************************************\n')
print('  1 -- Find initial signature genes (t-test)')
print('  2 -- Compute mean and std of top signature genes')
print('  3 -- Do cell type deconvolution')
print('\n*********************************************************\n')
choice = int(input("Input your choice: "))


stTime = time.mktime(time.gmtime())

Path = 'C:/Users/slu46/Documents/Research2023/ReDeconv_script'
#------ Input and output file name
fn_meta = Path + '/demo_data_4_deconvolution/References_Meta_data_subset.tsv'
fn_exp = Path + '/demo_data_4_deconvolution/References_scRNA_seq_Nor.tsv'


fn_ini_sig = './Results_4_deconvolution/Initial_sig_t_test_fd2.0_corr.tsv' 
fn_mean_std = './Results_4_deconvolution/Signature_mean_std_fd2.0.tsv'
fn_heatmap = './Results_4_deconvolution/Heatmap_signature_gene_matrix.png' #new file name
fn_extra_info = './Results_4_deconvolution/Signature_genes_extra_information.txt' #new file name


fn_bulk_RNAseq_raw = Path + './demo_data_4_deconvolution/Synthetic_Bulk_RNA_seq_Equal_Fraction_TPM.tsv'
fn_percentage_save ='./Results_4_deconvolution/ReDeconv_results.tsv'


if choice == 1:
   #Equivalent to Step-1 of GUI version for deconvolution !!!
   #Use t_test (pairwise, cell types) to find initial signature genes
   
   #Input: fn_meta, fn_exp
   #Parameter: L_max_pv, L_min_fold_change
   #Output: fn_ini_sig
   
   L_max_pv = 0.05
   L_min_fold_change = 2.0
   L_CellType_CellNo_LB = 30 #new parameter
   L_NoSep_sampleNo_UB = 2 #new parameter

   #Check if scRNA-seq and meta data are matched.
   L_status_data = check_meta_and_scRNAseq_data(fn_meta, fn_exp)
   
   if L_status_data>0:
      get_initial_Signature_Candidates(fn_meta, fn_exp, fn_ini_sig, L_max_pv, L_min_fold_change, L_CellType_CellNo_LB, L_NoSep_sampleNo_UB)
   
if choice == 2:
   #Equivalent to Step-2 of GUI version for deconvolution !!!
   #Get means and std for top signature genes chosen
   
   #Input: fn_meta, fn_exp, fn_ini_sig
   #Parameter: L_topNo
   #Output: fn_mean_std
   
   L_topNo = 133 #Upbound for number of signature genes for each cell type
   Get_signature_gene_matrix(fn_exp, fn_meta, fn_ini_sig, fn_mean_std, L_topNo, fn_heatmap, fn_extra_info)

if choice == 3:
   #Equivalent to Step-1 of GUI version for deconvolution !!!
   #Find percentages of cell types in mixture samples   
   
   #Input: fn_mean_std, fn_bulk_RNAseq_raw
   #Parameter: NA
   #Output: fn_percentage_save
  
   ReDeconv(fn_mean_std, fn_bulk_RNAseq_raw, fn_percentage_save)
     
endTime = time.mktime(time.gmtime())
print('\n\nTotal time =', ((endTime-stTime)/60), 'minutes')
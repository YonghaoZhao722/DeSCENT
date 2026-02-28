# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:42:42 2023

@author: Songjian Lu
"""
import time


from __ReDeconv_N import *



   
print('********************************************************\n')
print('  1 -- Compute gene expresion means for all cell types')
print('  2 -- Check the cell type classification quality for the scRNA-seq data (heatmap-plot)')
print('  3 -- Check the cell type classification quality for the scRNA-seq data (point-plot)')
print('  4 -- Do scRNA-seq data normalization for cell type deconvolution')
print('\n*********************************************************\n')
choice = int(input("Input your choice: "))

stTime = time.mktime(time.gmtime())


#------ Input file name
fn_meta = './demo_data_4_normalization/Demo_SN160K_5sp_meta.tsv'
fn_exp = './demo_data_4_normalization/Demo_SN160K_5sp_scRNAseq_2.tsv'

#------ Output file names for cell type count and mean information
fn_ctyp_mean = './Results_4_normalization/Ctype_size_means.tsv'
fn_ctyp_count = './Results_4_normalization/Ctype_cell_counts.tsv'
fn_cell_transcriptome_size = './Results_4_normalization/Cell_trans_sizes.tsv'


   
if choice == 1:
   #Compute gene expression means for all cell types
    
   '''
   In any sample, if the cell count of a cell type is less than "L_cell_count_Low_bound", 
   then the transcriptome size of the cell type in this sample would be set to "nan",
   which would not be used in the normalization. Default low bound is 10.
   The expresson profiles of those cells can still be normalized by using transcriptome size mean
   information of other cell types.
   '''
   #Input: fn_meta, fn_exp
   #Parameter: L_cell_count_Low_bound
   #Output: fn_ctyp_mean, fn_ctyp_count, fn_cell_transcriptome_size
   L_cell_count_Low_bound = 10
   
   L_status_data = check_meta_and_scRNAseq_data(fn_meta, fn_exp)
   if(L_status_data>0):
      get_sample_cell_type_exp_mean_and_cell_count(fn_meta, fn_exp, fn_ctyp_mean, fn_ctyp_count, fn_cell_transcriptome_size, L_cell_count_Low_bound)
   
if choice == 2:
   #Check if expression means of all cell types in any two sample have strong linear relation.
   #If the linear relation is not strong, it is likely that cells in each cell type are not purefied enough
   
 
   
   #------ output file names
   fn_heatmap = './Results_4_normalization/Heatmap_plot.png'
   fn_heatmap_matrix = './Results_4_normalization/Heatmap_plot_correlation_matrix.csv'
  


   #Draw heatmap of Pearson-correlation coeficients for all sample pairs
   #Input: fn_ctyp_mean
   #Output: fn_heatmap, fn_heatmap_matrix
   draw_heatmap_Pearson_all(fn_ctyp_mean, fn_heatmap, fn_heatmap_matrix)
   
   


if choice == 3:
   #Check if expression means of all cell types in any two sample have strong linear relation.
   #If the linear relation is not strong, it is likely that cells in each cell type are not purefied enough
   
 
   
   #------ output file names
   fn_extra_info = './Results_4_normalization/Extra_information.txt'
   fn_point = './Results_4_normalization/Points_plot.png'

   #----
   L_figureNo_eachRow = 2
   L_baseline = 1
   L_Pearson_LB = 0.999
   
   #This function can get the baseline sample such that we can merge the most number of samples together.
   #L_baseline = get_sample_baseline(fn_ctyp_mean, L_Pearson_LB)

   #Input: fn_ctyp_count, fn_ctyp_mean
   #Parameter: L_Pearson_LB
   #Output: fn_extra_info
   get_sample_cell_type_information_top_Pearson_2(fn_ctyp_count, fn_ctyp_mean, fn_extra_info, L_Pearson_LB)
   
   print('============================')

   #Input: fn_ctyp_mean
   #Parameters: L_baseline, L_Pearson_LB, L_figureNo_eachRow
   #Output: fn_point
   draw_cell_type_size_mean_point_plot(fn_ctyp_mean, fn_point, L_baseline, L_Pearson_LB, L_figureNo_eachRow)
   



if choice == 4:
   #Choose cells from some samples and perform normalization


   #File name of meta information for chosen cells. This file will not be created if all cells are chosen.
   fn_meta_2 = './Results_4_normalization/Meta_data_new.tsv'
   
   #File name for scRNA-seq data submatrix for chosen cells. This file will not be created if all cells are chosen.
   fn_exp_2 = './Results_4_normalization/scRNA_seq_temp_file.tsv'
   
   #File name for normalized scRNA-seq data 
   fn_exp_3 = './Results_4_normalization/scRNA_seq_new.tsv'

   
   L_Pearson_LB = 0.99
   L_baseline = 0

   #This function can get the baseline sample such that we can merge the most number of samples together.
   L_baseline = get_sample_baseline(fn_ctyp_mean, L_Pearson_LB)
   print('--Base line:', L_baseline)
   
   #Input: fn_exp, fn_meta, fn_ctyp_mean, fn_cell_transcriptome_size
   #Parameters: L_baseline, L_Pearson_LB
   #Output: fn_exp_2, fn_meta_2, fn_exp_3
   get_cell_subset_scRNA_seq_data_normalization(fn_exp, fn_meta, fn_ctyp_mean, fn_cell_transcriptome_size, fn_exp_2, fn_meta_2, fn_exp_3, L_baseline, L_Pearson_LB)

 
   
endTime = time.mktime(time.gmtime())
print('\n\nTotal time =', ((endTime-stTime)/60), 'minutes')
   
   

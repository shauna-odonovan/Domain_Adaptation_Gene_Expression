import pickle
import numpy as np
# from src.uda.DataHandler import ProcessData
from sklearn import svm
from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans
from sklearn.datasets import make_blobs
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


import sys
import os
import math

experiment ='GTX_C_pred_dom_720LE_L1O_matched'

compound_1='HCB_C'
compound_2='OPZ_C'
gene=48
first=gene*5
last_vivo=first+4
last_vitro=first+3

font_T=18
font_L=18


plot_name='plots'
extensions = '.p'

c1=[0,0.4470,0.7410]
c2=[0.8500,0.3250,0.0980]
c3=[0.9290, 0.6940, 0.1250]

f,((ax1,ax3,ax5,ax7,ax9,ax11),(ax2,ax4,ax6,ax8,ax10,ax12)) = plt.subplots(2,6,figsize=(20,10),dpi=900,sharex=True, sharey=True)

folder = 'export/' + experiment
contents = {}
for subdir, dirs, files in os.walk(folder):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext == extensions:
            file_path = os.path.join(subdir, file)
            # print(file_path)
            content = pickle.load(open(file_path, "rb"))
            # print(len(content))
            content_dict = {
                 'source_x_train': content[0],
                'target_x_train': content[1],
                'source_x_valid': content[2],
                'target_x_valid': content[3],
                'source_y_train': content[4],
                'source_y_valid': content[5],
                'training_mse': content[6],
                'validation_mse': content[7],
                'embeddings_e1': content[8],
                'embeddings_e2': content[9],
                'embeddings': content[10],
                'predictions': content[11],
                'all_usable_genes': content[12],
                'labels_train': content[13],
                'labels_valid': content[14],
                'tox_labels_train': content[15],
                'tox_labels_valid': content[16]
            }
            contents[content_dict['labels_valid'][0]] = content_dict
            
compound=contents[compound_1]
           
source_x_valid = compound['source_x_valid']
source_y_valid = compound['source_y_valid']
target_x_valid = compound['target_x_valid']
source_pred = compound['predictions']['source_valid']
target_pred = compound['predictions']['target_valid']


source_y_valid_nones = np.full((source_y_valid.shape[0],
                                        source_y_valid.shape[1] + int(
                                            source_y_valid.shape[1] / 4) - 1), None)
source_x_valid_nones = np.full(source_y_valid_nones.shape, None)
target_x_valid_nones = np.full(source_y_valid_nones.shape, None)

source_pred_valid_nones = np.full(source_y_valid_nones.shape, None)
target_pred_valid_nones = np.full(source_y_valid_nones.shape, None)

for i in range(0, source_x_valid.shape[1], 3):
    a = i + (int(i / 3) * 2)
    # print(data_handler.get_source_x_train_original().shape)
    source_x_valid_nones[:, a:a + 3] = source_x_valid[:, i:i + 3]
    target_x_valid_nones[:, a:a + 3] = target_x_valid[:, i:i + 3]

for i in range(0, source_y_valid.shape[1], 4):
    a = i + int(i / 4)

    source_y_valid_nones[:, a:a + 4] = source_y_valid[:, i:i + 4]
    # print(predictions['source_train'].shape)
    source_pred_valid_nones[:, a:a + 4] = source_pred[:, i:i + 4]
    target_pred_valid_nones[:, a:a + 4] = target_pred[:, i:i + 4]

    

 # source_x_valid_c = source_x_valid_nones[:8]
source_x_valid_c_1 = source_x_valid_nones[0]
source_x_valid_c_2 = source_x_valid_nones[2]



    # source_x_valid_l = source_x_valid_nones[8:16]
source_x_valid_l_1 = source_x_valid_nones[4]
source_x_valid_l_2 = source_x_valid_nones[6]

    # source_x_valid_m = source_x_valid_nones[16:24]
source_x_valid_m_1 = source_x_valid_nones[8]
source_x_valid_m_2 = source_x_valid_nones[10]

    # source_x_valid_h = source_x_valid_nones[24:32]
source_x_valid_h_1 = source_x_valid_nones[12]
source_x_valid_h_2 = source_x_valid_nones[14]

    # source_y_valid_c = source_y_valid_nones[:8]
source_y_valid_c_1 = source_y_valid_nones[0]
source_y_valid_c_2 = source_y_valid_nones[1]


    # source_y_valid_l = source_y_valid_nones[8:16]
source_y_valid_l_1 = source_y_valid_nones[4]
source_y_valid_l_2 = source_y_valid_nones[5]

    # source_y_valid_m = source_y_valid_nones[16:24]
source_y_valid_m_1 = source_y_valid_nones[8]
source_y_valid_m_2 = source_y_valid_nones[9]

    # source_y_valid_h = source_y_valid_nones[24:32]
source_y_valid_h_1 = source_y_valid_nones[12]
source_y_valid_h_2 = source_y_valid_nones[13]

    # target_x_valid_c = target_x_valid_nones[:8]
target_x_valid_c_1 = target_x_valid_nones[0]
target_x_valid_c_2 = target_x_valid_nones[2]


    # target_x_valid_l = target_x_valid_nones[8:16]
target_x_valid_l_1 = target_x_valid_nones[4]
target_x_valid_l_2 = target_x_valid_nones[6]

# target_x_valid_m = target_x_valid_nones[16:24]
target_x_valid_m_1 = target_x_valid_nones[8]
target_x_valid_m_2 = target_x_valid_nones[10]

    # target_x_valid_h = target_x_valid_nones[24:32]
target_x_valid_h_1 = target_x_valid_nones[12]
target_x_valid_h_2 = target_x_valid_nones[14]

    # source_predict_c = source_pred_valid_nones[:8]
source_predict_c_1 = source_pred_valid_nones[0]
source_predict_c_2 = source_pred_valid_nones[2]

    # source_predict_l = source_pred_valid_nones[8:16]
source_predict_l_1 = source_pred_valid_nones[4]
source_predict_l_2 = source_pred_valid_nones[6]

# source_predict_m = source_pred_valid_nones[16:24]
source_predict_m_1 = source_pred_valid_nones[8]
source_predict_m_2 = source_pred_valid_nones[10]

    # source_predict_h = source_pred_valid_nones[24:32]
source_predict_h_1 = source_pred_valid_nones[12]
source_predict_h_2 = source_pred_valid_nones[14]

    # target_predict_c = target_pred_valid_nones[:8]
target_predict_c_1 = target_pred_valid_nones[0]
target_predict_c_2 = target_pred_valid_nones[2]

    # target_predict_l = target_pred_valid_nones[8:16]
target_predict_l_1 = target_pred_valid_nones[4]
target_predict_l_2 = target_pred_valid_nones[6]

# target_predict_m = target_pred_valid_nones[16:24]
target_predict_m_1 = target_pred_valid_nones[8]
target_predict_m_2 = target_pred_valid_nones[10]

# target_predict_h = target_pred_valid_nones[24:32]
target_predict_h_1 = target_pred_valid_nones[12]
target_predict_h_2 = target_pred_valid_nones[14]           


ax1.plot([1,6,12],source_x_valid_l_1[first:last_vitro],color=c2,marker='o',linewidth=2)
ax1.plot([1,4,8,12],source_y_valid_l_1[first:last_vivo],color=c1,marker='o',linewidth=2)
ax1.plot([1,4,8,12],source_y_valid_l_2[first:last_vivo],color=c1,marker='o',linewidth=2)
ax1.plot([1,4,8,12],source_predict_l_1[first:last_vivo],color=c3,marker='o',linewidth=2)
ax1.set_title(compound_1[:-2] +'\n low  r1',multialignment='center',fontsize=font_T)
ax1.set_ylabel('rat gene expression level',fontsize=font_L)


ax2.plot([1,6,12],target_x_valid_l_1[first:last_vitro],color=c2,marker='o',linewidth=2)
ax2.plot([1,4,8,12],target_predict_l_1[first:last_vivo],color=c3,marker='o',linewidth=2)
ax2.set_ylabel('human gene expression level',fontsize=font_L)



ax3.plot([1,6,12],source_x_valid_l_2[first:last_vitro],color=c2,marker='o',linewidth=2)
ax3.plot([1,4,8,12],source_y_valid_l_1[first:last_vivo],color=c1,marker='o',linewidth=2)
ax3.plot([1,4,8,12],source_y_valid_l_2[first:last_vivo],color=c1,marker='o',linewidth=2)
ax3.plot([1,4,8,12],source_predict_l_2[first:last_vivo],color=c3,marker='o',linewidth=2)
ax3.set_title(compound_1[:-2] +'\n low r2',multialignment='center',fontsize=font_T)

ax4.plot([1,6,12],target_x_valid_m_2[first:last_vitro],color=c2,marker='o',linewidth=2)
ax4.plot([1,4,8,12],target_predict_m_2[first:last_vivo],color=c3,marker='o',linewidth=2)

ax5.plot([1,6,12],source_x_valid_m_2[first:last_vitro],color=c2,marker='o',linewidth=2)
ax5.plot([1,4,8,12],source_y_valid_m_1[first:last_vivo],color=c1,marker='o',linewidth=2)
ax5.plot([1,4,8,12],source_y_valid_m_2[first:last_vivo],color=c1,marker='o',linewidth=2)
ax5.plot([1,4,8,12],source_predict_m_2[first:last_vivo],color=c3,marker='o',linewidth=2)
ax5.set_title(compound_1[:-2] +'\n medium r2',multialignment='center',fontsize=font_T)

ax6.plot([1,6,12],target_x_valid_m_2[first:last_vitro],color=c2,marker='o',linewidth=2)
ax6.plot([1,4,8,12],target_predict_m_2[first:last_vivo],color=c3,marker='o',linewidth=2)


compound=contents[compound_2]

source_x_valid = compound['source_x_valid']
source_y_valid = compound['source_y_valid']
target_x_valid = compound['target_x_valid']
source_pred = compound['predictions']['source_valid']
target_pred = compound['predictions']['target_valid']

source_y_valid_nones = np.full((source_y_valid.shape[0],
                                        source_y_valid.shape[1] + int(
                                            source_y_valid.shape[1] / 4) - 1), None)
source_x_valid_nones = np.full(source_y_valid_nones.shape, None)
target_x_valid_nones = np.full(source_y_valid_nones.shape, None)

source_pred_valid_nones = np.full(source_y_valid_nones.shape, None)
target_pred_valid_nones = np.full(source_y_valid_nones.shape, None)

for i in range(0, source_x_valid.shape[1], 3):
    a = i + (int(i / 3) * 2)
    # print(data_handler.get_source_x_train_original().shape)
    source_x_valid_nones[:, a:a + 3] = source_x_valid[:, i:i + 3]
    target_x_valid_nones[:, a:a + 3] = target_x_valid[:, i:i + 3]

for i in range(0, source_y_valid.shape[1], 4):
    a = i + int(i / 4)
    source_y_valid_nones[:, a:a + 4] = source_y_valid[:, i:i + 4]
    # print(predictions['source_train'].shape)
    source_pred_valid_nones[:, a:a + 4] = source_pred[:, i:i + 4]
    target_pred_valid_nones[:, a:a + 4] = target_pred[:, i:i + 4]

# source_x_valid_c = source_x_valid_nones[:8]
source_x_valid_c_1 = source_x_valid_nones[0]
source_x_valid_c_2 = source_x_valid_nones[2]



    # source_x_valid_l = source_x_valid_nones[8:16]
source_x_valid_l_1 = source_x_valid_nones[4]
source_x_valid_l_2 = source_x_valid_nones[6]

    # source_x_valid_m = source_x_valid_nones[16:24]
source_x_valid_m_1 = source_x_valid_nones[8]
source_x_valid_m_2 = source_x_valid_nones[10]

    # source_x_valid_h = source_x_valid_nones[24:32]
source_x_valid_h_1 = source_x_valid_nones[12]
source_x_valid_h_2 = source_x_valid_nones[14]

    # source_y_valid_c = source_y_valid_nones[:8]
source_y_valid_c_1 = source_y_valid_nones[0]
source_y_valid_c_2 = source_y_valid_nones[1]

    # source_y_valid_l = source_y_valid_nones[8:16]
source_y_valid_l_1 = source_y_valid_nones[4]
source_y_valid_l_2 = source_y_valid_nones[5]

    # source_y_valid_m = source_y_valid_nones[16:24]
source_y_valid_m_1 = source_y_valid_nones[8]
source_y_valid_m_2 = source_y_valid_nones[9]

    # source_y_valid_h = source_y_valid_nones[24:32]
source_y_valid_h_1 = source_y_valid_nones[12]
source_y_valid_h_2 = source_y_valid_nones[13]

    # target_x_valid_c = target_x_valid_nones[:8]
target_x_valid_c_1 = target_x_valid_nones[0]
target_x_valid_c_2 = target_x_valid_nones[2]

    # target_x_valid_l = target_x_valid_nones[8:16]
target_x_valid_l_1 = target_x_valid_nones[4]
target_x_valid_l_2 = target_x_valid_nones[6]

# target_x_valid_m = target_x_valid_nones[16:24]
target_x_valid_m_1 = target_x_valid_nones[8]
target_x_valid_m_2 = target_x_valid_nones[10]

    # target_x_valid_h = target_x_valid_nones[24:32]
target_x_valid_h_1 = target_x_valid_nones[12]
target_x_valid_h_2 = target_x_valid_nones[14]

    # source_predict_c = source_pred_valid_nones[:8]
source_predict_c_1 = source_pred_valid_nones[0]
source_predict_c_2 = source_pred_valid_nones[2]

    # source_predict_l = source_pred_valid_nones[8:16]
source_predict_l_1 = source_pred_valid_nones[4]
source_predict_l_2 = source_pred_valid_nones[6]

# source_predict_m = source_pred_valid_nones[16:24]
source_predict_m_1 = source_pred_valid_nones[8]
source_predict_m_2 = source_pred_valid_nones[10]

    # source_predict_h = source_pred_valid_nones[24:32]
source_predict_h_1 = source_pred_valid_nones[12]
source_predict_h_2 = source_pred_valid_nones[14]

    # target_predict_c = target_pred_valid_nones[:8]
target_predict_c_1 = target_pred_valid_nones[0]
target_predict_c_2 = target_pred_valid_nones[2]

    # target_predict_l = target_pred_valid_nones[8:16]
target_predict_l_1 = target_pred_valid_nones[4]
target_predict_l_2 = target_pred_valid_nones[6]

# target_predict_m = target_pred_valid_nones[16:24]
target_predict_m_1 = target_pred_valid_nones[8]
target_predict_m_2 = target_pred_valid_nones[10]

# target_predict_h = target_pred_valid_nones[24:32]
target_predict_h_1 = target_pred_valid_nones[12]
target_predict_h_2 = target_pred_valid_nones[14]           


ax7.plot([1,6,12],source_x_valid_m_1[first:last_vitro],color=c2,marker='o',linewidth=2)
ax7.plot([1,4,8,12],source_y_valid_m_1[first:last_vivo],color=c1,marker='o',linewidth=2)
ax7.plot([1,4,8,12],source_y_valid_m_2[first:last_vivo],color=c1,marker='o',linewidth=2)
ax7.plot([1,4,8,12],source_predict_m_1[first:last_vivo],color=c3,marker='o',linewidth=2)
ax7.set_title(compound_2[:-2] +'\n medium r1',multialignment='center',fontsize=font_T)


ax8.plot([1,6,12],target_x_valid_m_1[first:last_vitro],color=c2,marker='o',linewidth=2)
ax8.plot([1,4,8,12],target_predict_m_1[first:last_vivo],color=c3,marker='o',linewidth=2)



rat_1,=ax8.plot([],[],color=c2,marker='o',linewidth=1.5,label='measured rat in vitro')
rat_2,=ax8.plot([],[],color=c1,marker='o',linewidth=1.5,label='measured rat in vivo replicates')
rat_3,=ax8.plot([],[],color=c3,marker='o',linewidth=1.5,label='predicted rat in vivo')


plt.figlegend(handles=[rat_1,rat_2,rat_3],bbox_to_anchor=(0.88,0.53),
           ncol=3,fontsize=18)


ax9.plot([1,6,12],source_x_valid_h_2[first:last_vitro],color=c2,marker='o',linewidth=2)
ax9.plot([1,4,8,12],source_y_valid_h_1[first:last_vivo],color=c1,marker='o',linewidth=2)
ax9.plot([1,4,8,12],source_y_valid_h_2[first:last_vivo],color=c1,marker='o',linewidth=2)
ax9.plot([1,4,8,12],source_predict_h_2[first:last_vivo],color=c3,marker='o',linewidth=2)
ax9.set_title(compound_2[:-2] +'\n medium r2',multialignment='center',fontsize=font_T)


ax10.plot([1,6,12],target_x_valid_h_2[first:last_vitro],color=c2,marker='o',linewidth=2)
ax10.plot([1,4,8,12],target_predict_h_2[first:last_vivo],color=c3,marker='o',linewidth=2)

human_1,=ax8.plot([],[],color=c2,marker='o',linewidth=1.5,label='measured human in vitro')
humna_2,=ax8.plot([],[],color=c3,marker='o',linewidth=1.5,label='predicted human in vivo')

ax11.plot([1,6,12],source_x_valid_h_2[first:last_vitro],color=c2,marker='o',linewidth=2)
ax11.plot([1,4,8,12],source_y_valid_h_1[first:last_vivo],color=c1,marker='o',linewidth=2)
ax11.plot([1,4,8,12],source_y_valid_h_2[first:last_vivo],color=c1,marker='o',linewidth=2)
ax11.plot([1,4,8,12],source_predict_h_2[first:last_vivo],color=c3,marker='o',linewidth=2)
ax11.set_title(compound_2[:-2] +'\n high r1',multialignment='center',fontsize=font_T)

ax12.plot([1,6,12],target_x_valid_h_2[first:last_vitro],color=c2,marker='o',linewidth=2)
ax12.plot([1,4,8,12],target_predict_h_2[first:last_vivo],color=c3,marker='o',linewidth=2)

plt.setp(ax2.get_xticklabels(),visible=False)
plt.setp(ax4.get_xticklabels(),visible=False)
plt.setp(ax6.get_xticklabels(),visible=False)
plt.setp(ax8.get_xticklabels(),visible=False)
plt.setp(ax10.get_xticklabels(),visible=False)
plt.setp(ax12.get_xticklabels(),visible=False)

plt.figlegend(handles=[human_1,humna_2],bbox_to_anchor=(0.77,0.1),
           ncol=2,fontsize=18)


plt.savefig('plots/manuscript_figures/human_gene_expression_'+compound_1 + compound_2+'.tif')

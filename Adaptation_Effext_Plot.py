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
plt.rcParams.update({'font.size': 18})

font_L=18
font_T=20

import sys
import os
import math

name='GTX_C_pred_dom_720LE_L1O_matched'

name_adapt = 'GTX_C_pred_dom_720LE_L1O_matched/predictions_23.p'
name_none = 'GTX_C_pred_720LE_L1O_matched/predictions_23.p'



plot_folder = 'plots/manuscript_figures/'
extensions = '.p'



MSE_train=[]
rat_domain_train=[]
human_domain_train=[]
MSE_valid=[]
rat_domain_valid=[]
human_domain_valid=[]
c1=[0,0.4470,0.7410]
c2=[0.8500,0.3250,0.0980]
c3=[0.9290, 0.6940, 0.1250]
           
cmap = cm.tab20c(np.linspace(0, 1, 20))
colours = [cmap[0], cmap[4], cmap[10]]
edge_c=['b',c2,c3]
#colours = [c2,c1]
markers=['o','^']

f,((ax1,ax2) ,(ax3,ax4) , (ax5,ax6))=plt.subplots(3,2,figsize=(15,17),dpi=900)

file_path = 'export/' + name_none 
contents = {}
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
    'embeddings_latent': content[10],
    'predictions': content[11],
    'all_usable_genes': content[12],
    'labels_train': content[13],
    'labels_valid': content[14],
    'tox_labels_train': content[15],
    'tox_labels_valid': content[16]
}



          
            
embeddings_l1=content_dict['embeddings_e1']  
rnd_train_indices = range(len(embeddings_l1['source_train']))
rnd_valid_indices = range(len(embeddings_l1['source_valid']))
embedding_array = np.concatenate((embeddings_l1['source_train'][rnd_train_indices],embeddings_l1['target_train'][rnd_train_indices],embeddings_l1['source_valid'][rnd_valid_indices],embeddings_l1['target_valid'][rnd_valid_indices]))
dlabels = [0 for i in embeddings_l1['source_train'][:,0]] + [1 for i in embeddings_l1['target_train'][:,0]]+[0 for i in embeddings_l1['source_valid'][:,0]] +[1 for i in embeddings_l1['target_valid'][:,0]]

pca=PCA(3)
projected=pca.fit_transform(embedding_array)

x=projected[:,0]
y=projected[:,1]
z=projected[:,2]

explained_v = pca.explained_variance_ratio_

#ax = Axes3D(fig)
for i in range(0,len(projected[:,0])):
        ax1.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax1.set_xlabel('PC 1 : '+ str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=font_L)
ax1.set_ylabel('Layer 1 [135 nodes] \n \n PC 2 : '+ str(round((100*explained_v[1]),1)) + '% explained variance',multialignment='center',fontsize=font_L)
ax1.set_title('Without Domain Adaptation',fontsize=font_T)

embeddings_l2=content_dict['embeddings_e2']  
rnd_train_indices = range(len(embeddings_l2['source_train']))
rnd_valid_indices = range(len(embeddings_l2['source_valid']))
embedding_array = np.concatenate((embeddings_l2['source_train'][rnd_train_indices],embeddings_l2['target_train'][rnd_train_indices],embeddings_l2['source_valid'][rnd_valid_indices],embeddings_l2['target_valid'][rnd_valid_indices]))
dlabels = [0 for i in embeddings_l2['source_train'][:,0]] + [1 for i in embeddings_l2['target_train'][:,0]]+[0 for i in embeddings_l2['source_valid'][:,0]] +[1 for i in embeddings_l2['target_valid'][:,0]]

pca=PCA(3)
projected=pca.fit_transform(embedding_array)

x=projected[:,0]
y=projected[:,1]
z=projected[:,2]

explained_v = pca.explained_variance_ratio_


#ax = Axes3D(fig)
for i in range(0,len(projected[:,0])):
        ax3.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax3.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=font_L)
ax3.set_ylabel('Layer 2 [95 nodes] \n \n PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=font_L)
#ax.set_zlabel('PC 3')





embeddings_latent=content_dict['embeddings_latent']  
rnd_train_indices = range(len(embeddings_latent['source_train']))
rnd_valid_indices = range(len(embeddings_latent['source_valid']))
embedding_array = np.concatenate((embeddings_latent['source_train'][rnd_train_indices],embeddings_latent['target_train'][rnd_train_indices],embeddings_latent['source_valid'][rnd_valid_indices],embeddings_latent['target_valid'][rnd_valid_indices]))
dlabels = [0 for i in embeddings_latent['source_train'][:,0]] + [1 for i in embeddings_latent['target_train'][:,0]]+[0 for i in embeddings_latent['source_valid'][:,0]] +[1 for i in embeddings_latent['target_valid'][:,0]]

pca=PCA(3)
projected=pca.fit_transform(embedding_array)

x=projected[:,0]
y=projected[:,1]
z=projected[:,2]

explained_v = pca.explained_variance_ratio_


#ax = Axes3D(fig)
for i in range(0,len(projected[:,0])):
        ax5.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax5.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=font_L)
ax5.set_ylabel('Layer 3 [64 nodes]  \n \nPC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=font_L)
#ax.set_zlabel('PC 3')



file_path = 'export/' + name_adapt
contents = {}
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
    'embeddings_latent': content[10],
    'predictions': content[11],
    'all_usable_genes': content[12],
    'labels_train': content[13],
    'labels_valid': content[14],
    'tox_labels_train': content[15],
    'tox_labels_valid': content[16]
}

drug=content_dict['labels_valid'][0]  
          
            
embeddings_l1=content_dict['embeddings_e1']  
rnd_train_indices = range(len(embeddings_l1['source_train']))
rnd_valid_indices = range(len(embeddings_l1['source_valid']))
embedding_array = np.concatenate((embeddings_l1['source_train'][rnd_train_indices],embeddings_l1['target_train'][rnd_train_indices],embeddings_l1['source_valid'][rnd_valid_indices],embeddings_l1['target_valid'][rnd_valid_indices]))
dlabels = [0 for i in embeddings_l1['source_train'][:,0]] + [1 for i in embeddings_l1['target_train'][:,0]]+[0 for i in embeddings_l1['source_valid'][:,0]] +[1 for i in embeddings_l1['target_valid'][:,0]]

pca=PCA(3)
projected=pca.fit_transform(embedding_array)

x=projected[:,0]
y=projected[:,1]
z=projected[:,2]

explained_v = pca.explained_variance_ratio_

#ax = Axes3D(fig)
for i in range(0,len(projected[:,0])):
        ax2.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax2.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=font_L)
ax2.set_ylabel('PC 2 : '+ str(round((100*explained_v[1]),1)) + '% explained variance',multialignment='center',fontsize=font_L)
#ax.set_zlabel('PC 3')
ax2.set_title('With Domain Adaptation',fontsize=font_T)


embeddings_l2=content_dict['embeddings_e2']  
rnd_train_indices = range(len(embeddings_l2['source_train']))
rnd_valid_indices = range(len(embeddings_l2['source_valid']))
embedding_array = np.concatenate((embeddings_l2['source_train'][rnd_train_indices],embeddings_l2['target_train'][rnd_train_indices],embeddings_l2['source_valid'][rnd_valid_indices],embeddings_l2['target_valid'][rnd_valid_indices]))
dlabels = [0 for i in embeddings_l2['source_train'][:,0]] + [1 for i in embeddings_l2['target_train'][:,0]]+[0 for i in embeddings_l2['source_valid'][:,0]] +[1 for i in embeddings_l2['target_valid'][:,0]]

pca=PCA(3)
projected=pca.fit_transform(embedding_array)

x=projected[:,0]
y=projected[:,1]
z=projected[:,2]

explained_v = pca.explained_variance_ratio_

#ax = Axes3D(fig)
for i in range(0,len(projected[:,0])):
        ax4.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax4.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=font_L)
ax4.set_ylabel('PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=font_L)
#ax.set_zlabel('PC 3')


embeddings_latent=content_dict['embeddings_latent']  
rnd_train_indices = range(len(embeddings_latent['source_train']))
rnd_valid_indices = range(len(embeddings_latent['source_valid']))
embedding_array = np.concatenate((embeddings_latent['source_train'][rnd_train_indices],embeddings_latent['target_train'][rnd_train_indices],embeddings_latent['source_valid'][rnd_valid_indices],embeddings_latent['target_valid'][rnd_valid_indices]))
dlabels = [0 for i in embeddings_latent['source_train'][:,0]] + [1 for i in embeddings_latent['target_train'][:,0]]+[0 for i in embeddings_latent['source_valid'][:,0]] +[1 for i in embeddings_latent['target_valid'][:,0]]

pca=PCA(3)
projected=pca.fit_transform(embedding_array)

x=projected[:,0]
y=projected[:,1]
z=projected[:,2]

explained_v = pca.explained_variance_ratio_

#ax = Axes3D(fig)
for i in range(0,len(projected[:,0])):
        ax6.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8) 
        
ax6.scatter([],[],color=colours[0],marker=markers[0],label='rat data',edgecolors=edge_c[0])
ax6.scatter([],[],color=colours[1],marker=markers[1],label='human data',edgecolors=edge_c[1])
ax6.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=font_L)
ax6.set_ylabel('PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=font_L)
#ax.set_zlabel('PC 3')
plt.figlegend(loc='lower center',
           ncol=2,fontsize=22)

plt.savefig(plot_folder  +  'Figure_3.pdf')
plt.savefig(plot_folder  +  'Figure_3.eps')
plt.savefig(plot_folder  +  'Figure_3.tif')
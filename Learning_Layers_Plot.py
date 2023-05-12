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
plt.rcParams.update({'font.size':16})


fontsize_L=18
fontsize_T=20

import sys
import os
import math

name='GTX_C_pred_dom_720LE_L1O_matched'

name_1 = name + '/interval_0.0_23.p'
name_2 = name + '/interval_25.0_23.p'
name_3 = name + '/interval_50.0_23.p'
name_4 = name + '/predictions_23.p'



plot_folder = 'plots/manuscript_figures/'
extensions = '.p'



MSE_train=[]
rat_domain_train=[]
human_domain_train=[]
MSE_valid=[]
rat_domain_valid=[]
human_domain_valid=[]

#fig = plt.figure(num=None, figsize=(11,11),dpi=300)            
cmap = cm.tab20c(np.linspace(0, 1, 20))
colours = [cmap[0], cmap[4], cmap[10]]
edge_c=['b','r','g']
markers=['o','^']

#f,((ax1,ax4,ax7,ax10) ,(ax2,ax5,ax8,ax11), (ax3,ax6,ax9,ax12))=plt.subplots(3,4,figsize=(26,20),dpi=200)
f,((ax1,ax4,ax7) ,(ax2,ax5,ax8), (ax3,ax6,ax9))=plt.subplots(3,3,figsize=(20,19),dpi=800)

file_path = 'midterm/' + name_1 
contents = {}
content = pickle.load(open(file_path, "rb"))
            # print(len(content))
content_dict = {
    'training_mse': content[0],
    'validation_mse': content[1],
    'embeddings_e1': content[2],
    'embeddings_e2': content[3],
    'embeddings_latent': content[4],
    'tox_labels_train': content[5],
    'tox_labels_valid': content[6]
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
ax1.set_xlabel('PC 1 : '+ str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax1.set_ylabel('Layer 1 [135 nodes] \n \n PC 2 : '+ str(round((100*explained_v[1]),1)) + '% explained variance',multialignment='center',fontsize=fontsize_L)
ax1.set_title('No training',fontsize=fontsize_T)

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
        ax2.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax2.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax2.set_ylabel('Layer 2 [95 nodes] \n \n PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=fontsize_L)
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
        ax3.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)
ax3.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax3.set_ylabel('Layer 3 [64 nodes] \n \n PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=fontsize_L)
#ax.set_zlabel('PC 3')


file_path = 'midterm/' + name_2 
contents = {}
content = pickle.load(open(file_path, "rb"))
            # print(len(content))
content_dict = {
    'training_mse': content[0],
    'validation_mse': content[1],
    'embeddings_e1': content[2],
    'embeddings_e2': content[3],
    'embeddings_latent': content[4],
    'tox_labels_train': content[5],
    'tox_labels_valid': content[6]
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
        ax4.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax4.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax4.set_ylabel('PC 2 : '+ str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=fontsize_L)
ax4.set_title('25% point in training',fontsize=fontsize_T)
#ax.set_zlabel('PC 3')



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
        ax5.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax5.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax5.set_ylabel('PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=fontsize_L)
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
ax6.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax6.set_ylabel('PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=fontsize_L)
#ax.set_zlabel('PC 3')

file_path = 'midterm/' + name_3 
contents = {}
content = pickle.load(open(file_path, "rb"))
            # print(len(content))
content_dict = {
    'training_mse': content[0],
    'validation_mse': content[1],
    'embeddings_e1': content[2],
    'embeddings_e2': content[3],
    'embeddings_latent': content[4],
    'tox_labels_train': content[5],
    'tox_labels_valid': content[6]
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
        ax7.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax7.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax7.set_ylabel('PC 2 : '+ str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=fontsize_L)
ax7.set_title('mid-way point in training',fontsize=fontsize_T)
#ax.set_zlabel('PC 3')



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
        ax8.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)   
ax8.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax8.set_ylabel('PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=fontsize_L)
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
        ax9.scatter(x[i], y[i],c=colours[dlabels[i]], marker=markers[dlabels[i]],edgecolors=edge_c[dlabels[i]],alpha=0.8)  
rat_lab=ax9.scatter([],[],c=colours[0],marker=markers[0],edgecolor=edge_c[0],alpha=0.8, label='rat data')
human_lab=ax9.scatter([],[],c=colours[1],marker=markers[1],edgecolor=edge_c[1],alpha=0.8, label='human data')
ax9.set_xlabel('PC 1 : ' + str(round((100*explained_v[0]),1)) + '% explained variance',fontsize=fontsize_L)
ax9.set_ylabel('PC 2 : ' + str(round((100*explained_v[1]),1)) + '% explained variance',fontsize=fontsize_L)
#ax.set_zlabel('PC 3')




plt.figlegend(handles=[rat_lab,human_lab],loc='lower center',
           ncol=2,fontsize=20)

plt.savefig(plot_folder +'/' +  'PCA_learning_0_50'+'_HCB')
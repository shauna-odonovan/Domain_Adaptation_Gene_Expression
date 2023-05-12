import pickle
import numpy as np
# from src.uda.DataHandler import ProcessData
from sklearn import svm
from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans
from sklearn.datasets import make_blobs

import sys
import os
import math

MAKE_PREDICTION_PLOTS = False
MAKE_BOXPLOTS = False

name = 'GTX_C_pred_dom_720LE_L1O_matched'
folder = 'export/' + name + '/'
plot_folder = 'plots/' + name + '/'
extensions = '.p'

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
                'embeddings_latent': content[10],
                'predictions': content[11],
                'all_usable_genes': content[12],
                'labels_train': content[13],
                'labels_valid': content[14],
                'tox_labels_train': content[15],
                'tox_labels_valid': content[16]
            }
            contents[content_dict['labels_valid'][0]] = content_dict
def classify_svm(X_t, y_t,X_v,y_v):
    clf = svm.SVC(kernel='linear',C=1)
    clf.fit(X_t, y_t)
    pred = clf.predict(X_t)
    pred_valid=clf.predict(X_v)
    abs_train_total = np.mean(np.abs((y_t - pred)))
    abs_valid_total = np.mean(np.abs((y_v - pred_valid)))
        
    abs_valid_low=np.mean(np.abs((y_v[0:4]-pred_valid[0:4])))
    abs_valid_med=np.mean(np.abs((y_v[4:8]-pred_valid[4:8])))
    abs_valid_high=np.mean(np.abs((y_v[8:12]-pred_valid[8:12])))
                                  
    #print("mse_train:", abs_train_total,"mse_valid:",abs_valid_total)
    #print('MSE low : ',abs_valid_low)
    #print('MSE medium : ',abs_valid_med)
    #print('MSE high : ',abs_valid_high)
        #print( class_valid)
    return [abs_train_total,abs_valid_total,abs_valid_low,abs_valid_med,abs_valid_high,y_v,pred_valid]

def svm_class_GTX(compound):
    #print()
    #print(compound['labels_valid'][0])
    source_train=[]
    target_train=[]
    train_labels_GTX=[]
    #for rat in vitro data
    source_train_full = compound['embeddings_latent']['source_train']
    source_valid_full = compound['embeddings_latent']['source_valid']

    #for human in vitro data
    target_train_full = compound['embeddings_latent']['target_train']
    target_valid_full = compound['embeddings_latent']['target_valid']


    train_labels_GTX_full = compound['tox_labels_train'][:,0]
    
    
    valid_labels_GTX_full = compound['tox_labels_valid'][:,0]
    source_train=source_train_full[4:16,:]
    target_train=target_train_full[4:16,:]
    train_labels_GTX=train_labels_GTX_full[4:16]
    for i in range(1,44):
        low_ind=16*i+4
        high_ind=16*i+16
        source_train=np.concatenate((source_train,source_train_full[low_ind:high_ind,:]))
        target_train=np.concatenate((target_train,target_train_full[low_ind:high_ind,:]))
        train_labels_GTX=np.concatenate((train_labels_GTX,train_labels_GTX_full[low_ind:high_ind]))
    source_valid=source_valid_full[4:16]
    target_valid=target_valid_full[4:16]
    valid_labels_GTX=valid_labels_GTX_full[4:16]                                                 


    train_combi=np.concatenate((source_train,target_train))
    valid_combi=np.concatenate((source_valid,target_valid))
    
    train_labels=np.concatenate((train_labels_GTX,train_labels_GTX))
    valid_labels=np.concatenate((valid_labels_GTX,valid_labels_GTX))
    
    

    

    # source_train = target_x_train

    #print("\tSeparate GTX from not GTX")
    X_t = train_combi[np.where(train_labels != 0.5)].copy()
    y_t = train_labels[np.where(train_labels != 0.5)].copy()
    X_v = valid_combi.copy()
    y_v = valid_labels.copy()
    GnG = classify_svm(X_t, y_t,X_v,y_v)
    
    return GnG

def svm_class_C(compound):
    #print()
    #print(compound['labels_valid'][0])
    
    source_train=[]
    target_train=[]
    train_labels_C=[]
    #for rat in vitro data
    source_train_full = compound['embeddings_latent']['source_train']
    source_valid_full = compound['embeddings_latent']['source_valid']

    #for human in vitro data
    target_train_full = compound['embeddings_latent']['target_train']
    target_valid_full = compound['embeddings_latent']['target_valid']


    train_labels_C_full = compound['tox_labels_train'][:,1]
    
    
    valid_labels_C_full = compound['tox_labels_valid'][:,1]
    
    source_train=source_train_full[4:16]
    target_train=target_train_full[4:16]
    train_labels_C=train_labels_C_full[4:16]
    for i in range(1,44):
        low_ind=16*i+4
        high_ind=16*i+16
        source_train=np.concatenate((source_train,source_train_full[low_ind:high_ind,:]))
        target_train=np.concatenate((target_train,target_train_full[low_ind:high_ind,:]))
        train_labels_C=np.concatenate((train_labels_C,train_labels_C_full[low_ind:high_ind]))
    source_valid=source_valid_full[4:16]
    target_valid=target_valid_full[4:16]
    valid_labels_C=valid_labels_C_full[4:16]

    
    train_combi=np.concatenate((source_train,target_train))
    valid_combi=np.concatenate((source_valid,target_valid))
    
    train_labels=np.concatenate((train_labels_C,train_labels_C))
    valid_labels=np.concatenate((valid_labels_C,valid_labels_C))

    #print("\tSeparate C from not C")
    X_t = train_combi[np.where(train_labels != 0.5)].copy()
    y_t = train_labels[np.where(train_labels != 0.5)].copy()
    X_v = valid_combi.copy()
    y_v = valid_labels.copy()
    CnC = classify_svm(X_t, y_t,X_v,y_v)


    return CnC



# sys.exit()
err_GTX=[]
err_C= []
measured_GTX=[]
predicted_GTX=[]
measured_C=[]
predicted_C=[]
for k, v in contents.items():
    if v['tox_labels_valid'][1,0] !=0.5:
        out=svm_class_GTX(v)
        err_GTX.append(out[0:5])
        measured_GTX+=[i for i in out[5]]
        predicted_GTX+=[i for i in out[6]]
    if v['tox_labels_valid'][1,1]!=0.5:
        out=svm_class_C(v)
        err_C.append(out[0:5])
        measured_C+=[i for i in out[5]]
        predicted_C+=[i for i in out[6]]

from sklearn.metrics import confusion_matrix        


print('GTX prediction')
print(len(err_GTX))
print(1-np.mean(err_GTX,axis=0))

valid_accuracy=np.mean(np.abs([measured_GTX[i]-predicted_GTX[i] for i in range(0,len(measured_GTX))]))

print('valid accuracy : ',valid_accuracy)
CM_GTX=confusion_matrix(measured_GTX,predicted_GTX)

print('confusion matrix')
print(CM_GTX)

total_GTX=sum(sum(CM_GTX))
acc_GTX=(CM_GTX[0,0]+CM_GTX[1,1])/total_GTX
sensitivity_GTX=CM_GTX[0,0]/(CM_GTX[0,0]+CM_GTX[0,1])
specificity_GTX=CM_GTX[1,1]/(CM_GTX[1,0]+CM_GTX[1,1])

print('accuracy : ',acc_GTX)
print('sensitivity : ',sensitivity_GTX)
print('specificity : ',specificity_GTX)

print('C prediction')
print(len(err_C))
print(1-np.mean(err_C,axis=0))

valid_accuracy=np.mean(np.abs([measured_C[i]-predicted_C[i] for i in range(0,len(measured_C))]))
for i in range(0,len(measured_C[:])):
    print(measured_C[i], predicted_C[i],measured_C[i]-predicted_C[i])
CM_C=confusion_matrix(measured_C,predicted_C)

total_C=sum(sum(CM_C))
acc_C=(CM_C[0,0]+CM_C[1,1])/total_C
sensitivity_C=CM_C[0,0]/(CM_C[0,0]+CM_C[0,1])
specificity_C=CM_C[1,1]/(CM_C[1,0]+CM_C[1,1])
print('CM C')
print(CM_C)
print('accuracy : ',acc_C)
print('sensitivity : ',sensitivity_C)
print('specificity : ',specificity_C)

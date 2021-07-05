"""
Auteur: Benny
Date: Nov 2019
Modifié par : Camille Lhenry
Dernière mise à jour : 30/06/2021


Objectif : Script permettant de tester le réseau de neurones PointNet++ entraîné pour la segmentation sémantique d'un nuage de points


"""

# -----------------------------------------------------------------------------
# Import des librairies et chemins d'accès
# -----------------------------------------------------------------------------


import argparse
import os
from data_utils.dataloader_vigo_thermique_rgb import train_val_dataset, TestWholeScene # /!\ : Utiliser le script de pré-traitements correspondant à la base de données utilisées
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import progressbar
import sklearn
from sklearn.metrics import confusion_matrix

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# -----------------------------------------------------------------------------
# Initialisation des classes et assignation à un numéro 
# -----------------------------------------------------------------------------

classes = ['Facade', 'Fenetre','Porte', 'Végétation', 'Sol'] # /!\ à adapter selon la tâche effectuée
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


# -----------------------------------------------------------------------------
# Initialisation des arguments de segmentation (changer les valeurs par défauts pour modifier)
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]') 
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--npoint', type=int,  default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--block_size', type=int, default=12)
    parser.add_argument('--stride', type=int, default=6)
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b,n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


# -----------------------------------------------------------------------------
# Segmentation du nuage
# ----------------------------------------------------------------------------

import sklearn
from sklearn.metrics import confusion_matrix


args=parse_args()
    
NUM_CLASSES = 5 #/!\ : à adpater à la base de données utilisée
NUM_POINT = args.npoint
BATCH_SIZE = args.batch_size                      
root='data/CAM_VIGO_VEGE/'
block_size = args.block_size
stride = args.stride
    
        

experiment_dir = Path('./log/sem_seg/2021-07-05_14-15') #/!\ : remplacer par le dossier log utilisé où se trouve le réseau entraîné
str(experiment_dir)
print("\nLe dossier d'entraînement choisi est : ", str(experiment_dir))
MODEL = importlib.import_module(args.model)
classifier = MODEL.get_model(NUM_CLASSES).cuda()
checkpoint=torch.load(str(experiment_dir)+'/checkpoints/best_model.pth')
    
# Dossier où se trouvera le nuage de points segmenté
visual_dir = str(experiment_dir) + '/visual/'
visual_dir = Path(visual_dir)
visual_dir.mkdir(exist_ok=True)
    
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier = classifier.eval()

# Le nuage de points à segmenté doit se trouver dans le sous-dossier Test de la base de données  
TEST_DATASET_WHOLE_SCENE =  TestWholeScene(split='test', root=root, block_points=NUM_POINT, block_size=block_size, stride =stride, num_classe=NUM_CLASSES) 
scene_id = TEST_DATASET_WHOLE_SCENE.file_list#Nom du fichier a traiter
scene_id = [x[:-4] for x in scene_id]
num_batches = len(TEST_DATASET_WHOLE_SCENE)#Nombre de fichier 
    
total_seen_class = [0 for _ in range(NUM_CLASSES)]
total_correct_class = [0 for _ in range(NUM_CLASSES)]
total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    
# -----------------------------------------------------------------------------
# Début de la segmentation du nuage
# -----------------------------------------------------------------------------

print('\n----------------------------')
print('Début de la segmentation du nuage...\n')
    
for batch_idx in range(num_batches):
        
    print("Nom du nuage segmenté : ",scene_id[batch_idx])
        
    whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
    whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
    vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
    total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
        
    for _ in tqdm(range(args.num_votes), total=args.num_votes):
        scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
        num_blocks = scene_data.shape[0]
        s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))  # /!\ : changer le nombre de caractéristiques en fonction de la base de données (attention : ajouter les coordonnées normalisées en tant que caractéristiques)
                
        batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
        batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
        batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))
                
        for sbatch in range(s_batch_num):
            start_idx = sbatch * BATCH_SIZE
            end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
            batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
            batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
            batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
            batch_data[:, :, 3:9] /= 1.0
            
            torch_data = torch.Tensor(batch_data)
            torch_data= torch_data.float().cuda()
            torch_data = torch_data.transpose(2, 1)
            seg_pred, _ = classifier(torch_data)
            batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
                    
            vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],batch_pred_label[0:real_batch_size, ...],batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)
                    
            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

        
# -----------------------------------------------------------------------------
# Calcul des critères de qualité (IoU, mIoU, global accuracy)
# -----------------------------------------------------------------------------


iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
arr = np.array(total_seen_class_tmp)
tmp_iou = np.mean(iou_map[arr != 0])
print('\nMean IoU (mIoU) of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
print('----------------------------')

IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
iou_per_class_str = '------- IoU --------\n' # /!\ : Adapter le nom des classes en fonction de la tâche de segmentation
print('------- IoU --------')
print("IoU classe façade : ", round(iou_map[0],2))
print("IoU classe fenêtre : ", round(iou_map[1],2)) 
print("IoU classe porte : ",round(iou_map[2],2))
print("IoU classe végétation : ",round(iou_map[3],2))
print("IoU classe sol : ",round(iou_map[4],2))
print('----------------------------')
       

print('\neval whole scene point avg class acc: %f' % (
np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
print('eval whole scene point accuracy: %f' % (
np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
print('----------------------------')

       
        
# -----------------------------------------------------------------------------
# Matrice de confusion
# -----------------------------------------------------------------------------
print('------- Matrice de confusion --------')
a = confusion_matrix(whole_scene_label, pred_label)
a_norm = confusion_matrix(whole_scene_label, pred_label, normalize = 'true')
print("\nMatrice de confusion : \n",a)
print("\nMatrice de confusion normalisée : \n",a_norm)
        
# -----------------------------------------------------------------------------
# Export du résultat segmenté
# -----------------------------------------------------------------------------

points=whole_scene_data
sortie=np.insert(points,points.shape[1],pred_label, axis=1)
sortie = np.savetxt(str(visual_dir) + "\\resultat.txt", sortie)
    
print("Done!")
    


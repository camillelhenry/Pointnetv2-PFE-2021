"""
Auteur: Benny
Date: Nov 2019
Modifié par : Camille Lhenry
Dernière mise à jour : 30/06/2021


Objectif : Script permettant d'entraîner et tester le réseau de neurones PointNet++ pour la segmentation sémantique de nuages de points

Ce programme est découpé en plusieurs parties : 

I. Import des librairies et chemins d'accès aux données 
II. Initialisation des classes et assignation à un numéro 
III. Entraînement du réseau de neurones
    1. Début de l'entraînement 
    2. Evaluation du modèle sur les données de validation 
    3. Tracer des graphiques permettant de suivre la qualité de l'entraînement'
IV. Test du réseau de neurones 
    1. Début de la segmentation du nuage
    2. Calcul des critères de qualité
    3. Calcul des matrices de confusion 
    4. Export du résultat en fichier .txt

Les modifications à réaliser pour utiliser le réseau sur d'autres données sont notifiées par le signe /!\
Les explications des lignes de codes sont notifiées par le signe /?\
    
"""


# -----------------------------------------------------------------------------
# Import des librairies et chemins d'accès
# -----------------------------------------------------------------------------

import argparse
import os
from data_utils.dataloader_vigo_thermique import train_val_dataset, TestWholeScene # /!\ : Utiliser le script de pré-traitements correspondant à la base de données utilisées
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

classes = ['Facade', 'Fenetre','balcon', 'toit', 'végétation'] # /!\ : Changer en fonction des classes présentes dans la base de données
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


# -----------------------------------------------------------------------------
# Initialisation des arguments de l'entraînement 
# -----------------------------------------------------------------------------

# /?\ : Permet de régler les hyper-paramètres de l'entraînement, notamment les tailles des lots, taille des blocs, nombre de points par bloc et le nombre d'époques
# Changer les valeurs par défaut

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]') # Version de PointNet++
    parser.add_argument('--batch_size', type=int, default=15, help='Batch Size during training [default: 16]') # Taille des lots
    parser.add_argument('--epoch',  default=25, type=int, help='Epoch to run [default: 128]') # Nombre d'époques
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]') # Utilisation du GPU
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]') # Optimisation de la descente de gradient
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=4096, help='Point Number [default: 4096]') # Nombre de points par bloc
    parser.add_argument('--step_size', type=int,  default=1, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate segmentation scores with voting [default: 5]') # Nombre d'itération lors de la phase de test
    parser.add_argument('--block_size', type=int, default=10) # Taille des blocs
    parser.add_argument('--stride', type=int, default=5) # Recouvrement des blocs lors de la phase de test 
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    # /?\ : Cette fonction permet d'effectuer plusieurs fois la segmentation du réseau lors de la phase test, pour moyenner les prédictions. 
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b,n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


# -----------------------------------------------------------------------------
# Entrainement du réseau de neurones
# -----------------------------------------------------------------------------


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''Initialisation de la carte graphique'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''Création du dossier où sera stocké l entraînement : log-sem_seg-date'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/') # Dossier où seront stockés les paramètres du réseau lors de l'entraînement 
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/') # Dossier où sera stocké le journal de l'entraînement
    log_dir.mkdir(exist_ok=True)
    graph_dir = experiment_dir.joinpath('graphique/') # Dossier où seront stockés les graphiques relatifs à l'entraînement (fonction d'erreur...)
    graph_dir.mkdir(exist_ok=True)

    '''LOG : création du fichier journal '''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    root = 'data/VIGO_THERMIQUE/' # /!\ : Chemin d'accès vers le dossier où sont stockées les données d'entraînement, organisées en trois sous-dossiers (Entraînement, Test, Validation)
    NUM_CLASSES = 5 # /!\ : à adapter au nombre de classe de la base de données
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    block_size = args.block_size

     # /?\ : Téléchargement des données à partir de la base et pré-traitement par le dataloader
    print("Téléchargement des données d'entraînement ...")
    TRAIN_DATASET =  train_val_dataset(split='train', data_root=root, num_point=NUM_POINT, num_classe=NUM_CLASSES, block_size=block_size, sample_rate=1, transform=None) # Données d'entraînement
    print("Téléchargement des données de test ...")
    TEST_DATASET =  train_val_dataset(split='validation', data_root=root, num_point=NUM_POINT, num_classe=NUM_CLASSES, block_size=block_size, sample_rate=1, transform=None) # Données de validation 
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("Le nombre de données d'entraînement est : %d" % len(TRAIN_DATASET))
    log_string("Le nombre de données test est : %d" % len(TEST_DATASET))

    '''Téléchargement du modèle'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))
    
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.eval()
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    # /?\ : Utilisation d'un modèle pré-entraîné s'il en existe
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Utilisation d un modèle pré-entraîné')
    except:
        log_string('Pas de modèle pré-entrainé, début d un nouvel entraînement...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    # /?\ : choix de l'algorithme d'optimisation de la descente de gradient, selon l'hyper-paramètre renseigné
    if args.optimizer == 'Adam':
        print('Optimiseur Adam')
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        print('Optimiseur SGD')
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    # /?\ Initialisation des listes permettant de tracer les graphes
    acc=[]
    lossg=[]
    lossval=[]
    accval=[]
    
    # ------------------------------------------------------------------------
    # Début de l'entraînement
    # ------------------------------------------------------------------------
    
    for epoch in range(start_epoch,args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        print('Momentum actuel :%f' %momentum)
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9): # Pour tous les blocs créés par le dataloader sur les données d'entraînement
            points, target = data
            points = points.data.numpy()
            points[:,:, :3] = provider.rotate_point_cloud_z(points[:,:, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(),target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points) # Passage du bloc dans le réseau de neurones
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights) # Calcul de l'erreur de prédiction 
            loss.backward() # Propagation de cette erreur dans les couches du réseau 
            optimizer.step()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches)) # Calcul des critères de qualité (erreur de prédiction)
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))  # Calcul des critères de qualité (excatitude)
        acc.insert(epoch, (total_correct / float(total_seen)))
        lossg.insert(epoch, (loss_sum / num_batches))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        # --------------------------------------------------------------------
        # Evaluation du modèle sur les données de validation
        # --------------------------------------------------------------------

        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9): # Pour tous les blocs créés par le dataloader sur les données de validation
                points, target = data
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                classifier = classifier.eval() 
                seg_pred, trans_feat = classifier(points) # Passage du bloc dans le réseau de neurones
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights) # Calcul de l'erreur de prédiction (qui ne sera pas rétro-propagé dans les couches du réseau)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp
                
                # Calcul des critères de qualité 
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l) )
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) )
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) )
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)) # mIoU : mean Intersection-over-Union
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches))) # erreur moyenne sur les données de validation
            lossval.insert(epoch, (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen))) # exactitude moyenne sur les données de validation
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            iou_per_class_str = '------- IoU --------\n' # IoU par classe
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            accval.insert(epoch, (total_correct / float(total_seen)))
            
            # /?\ : Boucle permettant de sauvegarder les meilleurs paramètres (critère de sélection : meilleur mIoU)
            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1
      
    # Fin de l'entraînement 
    
    # ------------------------------------------------------------------------
    # Tracer des graphs  d'exactitude et de fonction d'erreur 
    # -------------------------------------------------------------------------

    # /?\ Exactitude sur les données d'entraînement et de validation
    plt.figure(1)
    plt.plot(acc, 'r', label ="training accuracy")
    plt.plot(accval, 'b', label = "validation accuracy")
    plt.xlablel='Epoch'
    plt.ylabel='Accuracy'
    plt.legend(loc="upper left")
    plt.title('Training and validation accuracy')
    plt.savefig(str(graph_dir) + "\\Training and validation accuracy")
    plt.close(1)
   
    # /?\ Erreur sur les données d'entraînement et de validation
    plt.figure(2)
    plt.plot(lossg, 'r', label = "training loss")
    plt.plot(lossval, 'b', label = "validation loss")
    plt.xlablel='Epoch'
    plt.ylabel='Loss'
    plt.title('Training and validation loss')
    plt.legend(loc="upper left")
    plt.savefig(str(graph_dir) + "\\Training and validation loss")
    plt.close(2)
    
    
    
    
# -----------------------------------------------------------------------------
# Test du réseau de neurones
# -----------------------------------------------------------------------------    
    

    args=parse_args() # Récupération des arguments
    
    NUM_CLASSES = 5 # /!\ : à adpater à la base de données utilisée
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size                      
    block_size = args.block_size
    stride = args.stride
    
        
    str(experiment_dir)
    print("\nLe dossier d'entraînement choisi est : ", str(experiment_dir)) # Dossier où se trouvent les paramètres du réseau entrainé
    MODEL = importlib.import_module(args.model) # Import de l'architecture
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint=torch.load(str(experiment_dir)+'/checkpoints/best_model.pth') # Téléchargement des meilleurs paramètres
    
    # Dossier où se trouvera le nuage de points segmenté
    visual_dir = str(experiment_dir) + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)
    
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    
    TEST_DATASET_WHOLE_SCENE =  TestWholeScene(split='test', root=root, block_points=NUM_POINT, block_size=block_size, stride =stride, num_classe=NUM_CLASSES) # Pré-traitement des données test
    scene_id = TEST_DATASET_WHOLE_SCENE.file_list
    scene_id = [x[:-4] for x in scene_id]
    num_batches = len(TEST_DATASET_WHOLE_SCENE)
    
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
        
        # /?\ Initialisation de listes pour le calcul des critères de qualité
        whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
        whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
        total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
        
        for _ in tqdm(range(args.num_votes), total=args.num_votes): # Pour chaque itération (hyper-paramètre vote)
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))  # /!\ : changer le nombre de caractéristiques en fonction de la base de données (attention : ajouter les coordonnées normalisées en tant que caractéristiques)
                
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))
                
                for sbatch in range(s_batch_num): # Pour chaque bloc créé par le dataloader
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
                    seg_pred, _ = classifier(torch_data) # Prédiction des classes de chaque point 
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
        iou_per_class_str = '------- IoU --------\n'
        # /!\ : modifier le nom des classes pour qu'elles correspondent à la base de données utilisée
        print('------- IoU --------')
        print("IoU classe façade : ", round(iou_map[0],2))
        print("IoU classe fenêtre : ", round(iou_map[1],2))
        print("IoU classe balcon : ",round(iou_map[2],2))
        print("IoU classe toit : ",round(iou_map[3],2))
        print("IoU classe végétation : ",round(iou_map[4],2))
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
    print("\nMatrice de confusion normalisée : \n",a_norm) # Matrise de confusion normalisée
        
# -----------------------------------------------------------------------------
# Export du résultat segmenté
# -----------------------------------------------------------------------------

    points=whole_scene_data
    sortie=np.insert(points,points.shape[1],pred_label, axis=1)
    sortie = np.savetxt(str(visual_dir) + "\\resultat.txt", sortie)
    
    print("Done!")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)


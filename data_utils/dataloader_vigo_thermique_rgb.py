"""
Auteur: Benny
Date: Nov 2019
Modifié par : Camille Lhenry
Dernière mise à jour : 30/06/2021


Objectif : Script permettant de pré-traiter les nuages de points avant passage dans le réseau de neurones

Les modifications à réaliser pour utiliser le réseau sur d'autres données sont notifiées par le signe /!\
Les explications des lignes de codes sont notifiées par le signe /?\
    
"""



# -----------------------------------------------------------------------------
# Import des librairies 
# -----------------------------------------------------------------------------


import os
import numpy as np
from torch.utils.data import Dataset
import progressbar
import progressbar


# -----------------------------------------------------------------------------
# Traitement des nuages de points d'entraînement et de validation
# -----------------------------------------------------------------------------

class train_val_dataset(Dataset):
    
    def __init__(self, split='train', data_root='data/VIGO_THERMIQUE/', num_point=1024, num_classe=4, test_area=5, block_size=10.0, sample_rate=0, transform=None):
        super().__init__()
        self.num_point = num_point # Définition du nombre de points dans un bloc
        self.block_size = block_size # Taille du bloc 
        self.transform = transform  
        if split =='train':
            data_root = data_root + 'Entrainement/' # Pour l'entrainement
        if split == "validation":
            data_root = data_root + 'Validation/' # Pour la validation
        if split =='test':
            data_root = data_root + 'Test/' # Pour le test de nuage individuel
        nuage =  sorted(os.listdir(data_root)) # Liste le répertoire contenant les noms des fichier texte des nuages de points 
        print("Les nom des nuages dans le dossier ", split, "sont : ")
        for i in range(len(nuage)):
            print("\n", nuage[i])
        
                
        nuage_split = [nuag for nuag  in nuage] 
        # Initialisation de listes pour le traitement des nuages de points
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        
        # --------------------------------------------------------------------
        # Création des poids associés aux classes
        # --------------------------------------------------------------------
        #/?\ Les poids associés aux classes sémantiques permettent de favoriser la prédiction des classes peu représentées dans la base de données
        
        labelweights = np.zeros(num_classe)
        
        for room_name in nuage_split:         
            '''
            Pour chaque nuage dans la base de données : 
                Récupération du chemin d'accès
                Téléchargement des données avec np.load()
                Mise en forme des données en points/labels
                Récupération des données pour calculer les poids associés aux labels et les coord_min et coord_max
            '''
            room_path = os.path.join(data_root, room_name) # Chemin du nuage
            room_data = np.loadtxt(room_path)  # Téléchargement du nuage 
            points, labels = room_data[:, 0:6], room_data[:, 6]  # Division caractéristique / label     /!\ à modifier selon le nombre de caractéristiques des nuages de la base 
            tmp, bins = np.histogram(labels, bins=num_classe) # Pour chaque classe, compte le nombre de points dans chaque classe
            labelweights += tmp 
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels) 
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max) 
            num_point_all.append(labels.size) # Nombre de points dans le nuage 
            
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)  # Poids associés aux labels, inversement proportionnel au nombre de points dans la classe
        print("\nLes poids associés aux classes sont : ", self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all) 
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point) # Nombre d'itération à effectuer en fonction du nombre de points fixé pour couvrir l'ensemble des points de la base de données
        room_idxs = [] # Chaque ligne correspond à une itération et sur chaque ligne se trouve le numéro du nuage traité par l'itération
        
        for index in range(len(nuage_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
          
        self.room_idxs = np.array(room_idxs)
      
        
        
    # -------------------------------------------------------------------------
    # Division des nuages de points en bloc de num_point points
    # -------------------------------------------------------------------------


    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx] # Numéro du nuage à diviser
        points = self.room_points[room_idx]   # Nombre de points * nb de caractéristiques
        labels = self.room_labels[room_idx]   # N Label du nuage 
        N_points = points.shape[0] # Nombre de points dans le nuage de points 
        
        """ Tant que : Vrai
                Initialisation aléatoire d'un point central 
                Création d'un rectangle de la taille "block_size" autour du centre
                Récupération de tous les point du nuage appartenant à ce rectangle
                Si le nombre de point dépasse 1024 : stop
        """

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0] # Minimum du bloc 
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0] # Maximum du bloc 
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]

            if point_idxs.size > 1024:
                break

        #Pour permettant d'obtenir des blocs de 4096 points
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False) # /?\ Si le point_idx est supérieur à num_point : sous-échantillonnage
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True) # /?\ Si le point_idx et inférieur à num_point : sur-échantillonnage

        # ---------------------------------------------------------------------
        # Normalisation du nuage de points
        # ---------------------------------------------------------------------
        
        self.selected_points = points[selected_point_idxs, :]  
        self.current_points = np.zeros((self.num_point, 9))  
       
        # /?\ Normalisation des coordonnées par le maximum de chaque nuage
        self.current_points[:, 6] = self.selected_points[:, 0] / self.room_coord_max[room_idx][0]   
        self.current_points[:, 7] = self.selected_points[:, 1] / self.room_coord_max[room_idx][1]   
        self.current_points[:, 8] = self.selected_points[:, 2] / self.room_coord_max[room_idx][2]
        
        
        # /?\ Normalisation dans la sphere unité (pour traitement dans les niveau d'abstraction de PointNet++)
        centroid = np.mean(self.selected_points[:, 0:3], axis=0)
        
        self.selected_points[:, 0] = self.selected_points[:, 0] - centroid[0]
        self.selected_points[:, 1] = self.selected_points[:, 1] - centroid[1]
        self.selected_points[:, 2] = self.selected_points[:, 2] - centroid[2]
        m = np.max(np.sqrt(np.sum(self.selected_points[:,0:3]**2, axis=1)))
        self.selected_points[:,0:3] = self.selected_points[:,0:3]/m 
           
        self.current_points[:, 0:6] = self.selected_points 
        self.current_labels = labels[selected_point_idxs]  
        
        if self.transform is not None:
            self.current_points, self.current_labels = self.transform(self.current_points, self.current_labels)
        return self.current_points, self.current_labels

    def __len__(self):
        return len(self.room_idxs)
    
    
# -----------------------------------------------------------------------------
# Traitement du nuage de points test
# -----------------------------------------------------------------------------

class TestWholeScene():
    
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=5, block_size=10, padding=0.001, num_classe=7):
        
        # Préparation du découpage en bloc du nuage de points entier       
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        
        if self.split == 'test':
            root = root + 'Test/' # Chemin d'accès du nuage de points à tester
            self.file_list = os.listdir(root)
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        
        for file in self.file_list:
            data = np.loadtxt(root + file) # Téléchargement du nuage
            points = data[:, :3] # Coordonnées X Y Z 
            
            self.scene_points_list.append(data[:, :6]) # Coordonnées + caractéristiques /!\ : à adapter au nombre de caractéristiques de la base de données
            self.semantic_labels_list.append(data[:, 6]) # Label      
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)#Liste des 
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

       #  /?\ Attribution d'un poids pour chaque classe inversement proportionel au nombre de points par classe 
        
        labelweights = np.zeros(num_classe)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(num_classe+1))
            self.scene_points_num.append(seg.shape[0]) 
            labelweights += tmp#Definie les poids
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print("\nLes poids associés aux labels sont : ", self.labelweights)

    def __getitem__(self, index):
        
        # /?\ Découpage du nuage de points test
        
        # Définition de la grille 
        point_set_ini = self.scene_points_list[index] 
        points = point_set_ini[:,:6] # /!\ : à adapter au nombre de caractéristiques de la base de données
        labels = self.semantic_labels_list[index] # Recupère les labels associés
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1) # Permet de définir le nombre de fois que le nuage de points est divisé en X selon la taille du bloc        
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1) # Même chose en y
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
       
        
        for index_y in (range(0, grid_y)):
            
            for index_x in range(0, grid_x):#Pour chaque pas de la grille en X et Y 

                s_x = coord_min[0] + index_x * self.stride # Min du bloc en X
                e_x = min(s_x + self.block_size, coord_max[0])# Max du bloc en X
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride # Min du bloc 
                e_y = min(s_y + self.block_size, coord_max[1])# Max du bloc 
                s_y = e_y - self.block_size
                # Récupération des points à l'intérieur du bloc
                point_idxs = np.where((points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (points[:, 1] <= e_y + self.padding))[0]
               
                if point_idxs.size == 0:
                    continue
                
                num_batch = int(np.ceil(point_idxs.size / self.block_points))# Nombre de bloc pour chaque pas de la grille
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True 
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace) 
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :] 
                
                # Normalisation des coordonnées selon le min et max
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
              
                #Normalisation dans la sphere unité
                centroid = np.mean(data_batch[:, 0:3], axis=0)     
                data_batch[:, 0] = data_batch[:, 0] - centroid[0]
                data_batch[:, 1] = data_batch[:, 1] - centroid[1]
                data_batch[:, 2] = data_batch[:, 2] - centroid[2]
                m = np.max(np.sqrt(np.sum(data_batch[:,0:3]**2, axis=1)))
                data_batch[:,0:3] = data_batch[:,0:3]/m 
              
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)# Regroupe les caractéristiques ( ex : X Y Z R G B Xnorm Ynorm Znorm)
                label_batch = labels[point_idxs].astype(int)# Recupère les label 
                batch_weight = self.labelweights[label_batch]# Recupère les poids 

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
           
  
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room
    
    
        ''' Retour de la fonction 
        data_room : découpage en grille du nuage de points test
                    Matrice de la forme [nombre de blocs de découpe, nombre de points par bloc, nombre de caractéristiques du nuage]
                    Exemple : [251 , 4096 , 9]
        
        label_room : label pour chaque point dans les blocs de découpe
                     Matrice de la forme [nombre de découpe, nombre de points]
                     Exemple : [251, 4096]
        
        sample_weight : poid pour chaque point dans le sblocs de découpe 
                        Matrice de la forme [nombre de découpe, nombre de points]
                        Exemple : [251, 4096]
        
        index_room : index de chaque point des blocs dans le nuage de point complet, permettant la reconstruction de la scène globale après
                    segmentation des blocs par le réseau 
        '''
        

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()
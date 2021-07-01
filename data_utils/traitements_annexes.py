"""
Date: 01/07/2021
Créé par : Camille Lhenry


Objectif : Script permettant d'éffectuer des traitements annexes sur les nuages de points pour leurs mises en formes (X Y Z Caractéristiques Labels)"""



#%% Librairies nécéssaires

import os 
import numpy as np
import progressbar

#%% Réduction des coordonnées du nuage aux centroïdes
def reduction_donnees(x):
    meanx = np.mean(x[:,0])
    meany = np.mean(x[:,1])
    meanz = np.mean(x[:,2])
    x[:,0]=x[:,0]-meanx
    x[:,1]=x[:,1]-meany
    x[:,2]=x[:,2]-meanz
    return x

    
#%% Programme permettant de normalisé les données et de les mettre sous la forme X Y Z R G B Label 
path=os.getcwd()
data_root = "D:/PFE Alteirac_Lhenry/Lhenry/Base de données/VIGO/VIGO_THERMIQUE/nuage_thermique_lab_vege_non_crop_traitement/" 
nuage_vigo = sorted(os.listdir(data_root))
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for nuage_name in nuage_vigo:
     nuage_path = os.path.join(data_root, nuage_name)
     data = np.loadtxt(nuage_path)
     dataxyz=reduction_donnees(data[:,:3])
     datalabel=data[:,6]
     dataRGB=data[:,3:6]
     datafinal=np.c_[dataxyz, dataRGB,datalabel]
     np.savetxt(nuage_path, datafinal, fmt=fmt)
     

#%% Programmes permettant de gérer des données à 9 caractéristiques  : X Y Z R G B Cuv Label 

data_root="D:/PFE Alteirac_Lhenry/Lhenry/Base de données/vigo_sva_cubature_traitement/"
nuage_vigo = sorted(os.listdir(data_root))
nuage_vigo = [nuage for nuage in nuage_vigo]

fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.5f \t %d'

for nuage_name in nuage_vigo:
    nuage_path = os.path.join(data_root, nuage_name)
    nuage_data_load = np.loadtxt(nuage_path)
    dataxyz=reduction_donnees(nuage_data_load[:,:3]) # décallage des coordonnées
    datalab=nuage_data_load[:,7] # récupération des labels
    datacuv = nuage_data_load[:,6]  
    dataRGB = nuage_data_load[:,3:6]
    datafinal=np.c_[dataxyz, dataRGB,datacuv,datalab]
    #os.chdir(path +'/Training')
    np.savetxt(nuage_path, datafinal, fmt=fmt)

#%% PGR permettant de normaliser et de  mettre sous la forme X Y Z R G B Label les données de la base Paris-rue-Monge 
"""
façade : 0
devanture : 1
toit : 2
fenetre : 3
balcon : 4
porte : 5
"""
path=os.getcwd()
data_root = "D:/PFE Alteirac_Lhenry/Lhenry/Base de données/Rue_Monge/Rue_Monge_XYZRGBL_rot/" 
nuage_vigo = sorted(os.listdir(data_root))
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for nuage_name in nuage_vigo:
     nuage_path = os.path.join(data_root, nuage_name)
     data = np.loadtxt(nuage_path)
     for i in range(data.shape[0]):
         if data[i,6] == 3:
            data[i,6] = 2
         if data[i,6] == 4:
            data[i,6] = 3
         if data[i,6] == 5:
            data[i,6] = 4
         if data[i,6] == 6:
            data[i,6] = 5
            
         
     dataxyz=reduction_donnees(data[:,:3])
     datalabel=data[:,6]
     dataRGB=data[:,3:6]
     
     datafinal=np.c_[dataxyz, dataRGB,datalabel]
     np.savetxt(nuage_path, datafinal, fmt=fmt)
     
#%% PGR permettant de normaliser les données et de les mettre sous la forme X Y Z R G B Label (MUSEE ZOO)
path=os.getcwd()
data_root = "D:/PFE Alteirac_Lhenry/Lhenry/Base de données/ZOO/A TRAITER/"
nuage_vigo = sorted(os.listdir(data_root))
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for nuage_name in nuage_vigo:
     nuage_path = os.path.join(data_root, nuage_name)
     data = np.loadtxt(nuage_path)
     #dataxyz=reduction_donnees(data[:,:3])
     dataxyz = data[:,0:3]
     datalabel=data[:,6]
     dataRGB=data[:,3:6]
     datafinal=np.c_[dataxyz, dataRGB,datalabel]
     np.savetxt(nuage_path, datafinal, fmt=fmt)
     


#%% Extraction du plan principale et supression de l'arrière des fenêtres dans la base du Musée Zoologique

import os
import numpy as np
from tqdm import tqdm



data_root = "D:/PFE Alteirac_Lhenry/Lhenry/Base de données/ZOO/SANS_RESIDU_FENETRE_TRAITEMENT/"
nuage_name = sorted(os.listdir(data_root))
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for nuage_name in tqdm(nuage_name):
    nuage_path = os.path.join(data_root, nuage_name)
    data = np.loadtxt(nuage_path)
    nuage_z = data[:,2]
    mean = np.mean(nuage_z)
    condition = np.where((nuage_z >= mean-2))
    nuage_z_restreint = nuage_z[condition[0]]
    nuage_traite = data[condition[0]]
    np.savetxt(nuage_path, nuage_traite, fmt = fmt)
    
#%% Export VIGO THERMIQUE/RGB/Musée Zoologique 3D FEAT

import os
import numpy as np
from tqdm import tqdm



data_root = "D:/PFE Alteirac_Lhenry/Lhenry/Base de données/ZOO/AVEC_RESIDU_FENETRE/Traitement/"
nuage_name = sorted(os.listdir(data_root))
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t%1.3f \t %1.3f \t %1.3f \t %d'

for nuage_name in tqdm(nuage_name):
    nuage_path = os.path.join(data_root, nuage_name)
    data = np.loadtxt(nuage_path)
    nuage_xyz = data[:,0:3]
    nuage_RGB = data[:,3:6]
    nuage_label = data[:,6]
    nuage_3Dfeat = data[:,7:10]
    for i in tqdm(range(data.shape[0])):
        if np.isnan(nuage_3Dfeat[i,0]) == True:
            nuage_3Dfeat[i,0] = 0
        if np.isnan(nuage_3Dfeat[i,1]) == True:
            nuage_3Dfeat[i,1] = 0
        if np.isnan(nuage_3Dfeat[i,2]) == True:
            nuage_3Dfeat[i,2] = 0 
    nuage_final=np.c_[nuage_xyz, nuage_RGB, nuage_3Dfeat, nuage_label]
    np.savetxt(nuage_path, nuage_final, fmt = fmt)
    
    
#%% Suppression des 3D Feature sur Musée zoo sans fenetre


import os
import numpy as np
from tqdm import tqdm

data_root = "D:/PFE Alteirac_Lhenry/Lhenry/Base de données/ZOO/SANS_RESIDU_FENETRE_OK/Traitement/"
nuage_name = sorted(os.listdir(data_root))
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for nuage_name in tqdm(nuage_name):
    nuage_path = os.path.join(data_root, nuage_name)
    data = np.loadtxt(nuage_path)
    nuage_xyz = data[:,0:3]
    nuage_RGB = data[:,3:6]
    nuage_label = data[:,9]
    nuage_3Dfeat = data[:6:9]
    nuage_final=np.c_[nuage_xyz, nuage_RGB, nuage_label]
    np.savetxt(nuage_path, nuage_final, fmt = fmt)
    
            
        
    



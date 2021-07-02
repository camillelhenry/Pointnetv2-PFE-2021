"""
Auteur: Benny
Date: Nov 2019
Commenté par : Camille Lhenry
Dernière mise à jour : 30/06/2021    

Objectif: script d'implémentation du modèle PointNet++ pour la segmentation sémantique (méthode de regroupement MSG)
"""


import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes): # Initialisation du modèle
        super(get_model, self).__init__()
    
        # Définition des niveaux d'abstraction de PointNet++
        """Arguments de la fonction : PointNetSetAbstractionMsg(nombre de centre, [rayon 1, rayon 2], [nb de points dans la sphère 1, nb de points dans la 
            sphère 2], nombre de caractéristiques du nuage, [définition du 1er MLP, définition du 2nd MLP])"""
    
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(256, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        
        # Propagation des caractéristiques à travers les couches du réseau 
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        # Réseau dense pour effectuer les prédictions
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz): # Passage des blocs de points dans le modèle
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module): # Calcul de l'erreur de prédiction
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))
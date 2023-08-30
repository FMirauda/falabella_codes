# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Importaciones
import torch
from torch import nn

# Modelo de clasificación 1
class MultilabelClassifier1(nn.Module):
    def __init__(self, n_classes, pretrain_model):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.model_wo_fc = nn.Sequential(*(list(self.pretrain_model.children())[:-1]))

        self.classes = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features=n_classes)
        )

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'label': self.classes(x)
        }
    
# Modelo de clasificación 2
class MultilabelClassifier2(nn.Module):
    def __init__(self, n_classes, pretrain_model):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.model_wo_fc = nn.Sequential(*(list(self.pretrain_model.children())[:-1]))

        self.classes = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features=3)
        )

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'label': self.classes(x)
        }

###########################################################################################################
# Importaciones dataset
from torchvision import transforms 
from torch.utils.data import Dataset
import torch
from skimage import transform

# Dataset class para generar los batches
class PredictionDataset(Dataset):
    """Photos prediction dataset."""

    def __init__(self, lista_fotos, normalization = 'mlc', device = 'cuda'):
        """
        Args:
            lista_fotos (list): Lista con varias fotos concatenadas y sus areas
            normalization (string): Indica si se debe implementar la transformación
            de normalización para el clasificador multi-etiqueta ('mlc') o para el 
            detector de objetos ('ob')
            device (string): Indica qué dispositivo se quiere usar: 'cuda' o 'cpu'
        
        Returns:
            pro_img (tensor): Retorna la imagen procesada como tensor lista para la predicción.
        """
        self.lista_fotos = lista_fotos
        self.normalization = normalization
        self.device = device
        # transformación para clasificador
        self.mlc_normalization = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])
    
    def __len__(self):
        return len(self.lista_fotos)

    def __getitem__(self, idx):
        foto = self.lista_fotos[idx][0]
        area = self.lista_fotos[idx][1]
        url = self.lista_fotos[idx][2]
        # aplicar normalizacion
        if self.normalization == 'mlc':
            pro_img  = self.mlc_normalization(foto)
            #print(t_img.shape)
            #pro_img = t_img.unsqueeze(dim = 0)
            #print(pro_img.shape)
        
        elif self.normalization == 'od':
            t_img_0 = transform.resize(foto, (480, 480))
            pro_img = torch.Tensor(t_img_0).permute(2,0,1)

        # retornar imagen procesada
        return pro_img, area, url

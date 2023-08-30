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
# Importaciones modelos
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

from google.cloud import storage
import numpy as np
import cv2

# Dataset class para generar los batches
class PredictionDataset(Dataset):
    """Photos prediction dataset."""

    def __init__(self, bucket_name, blob_name_list, normalization = 'mlc', device = 'cuda'):
        """
        Args:
            bucket_name (str): Nombre del bucket
            blob_name_list (list): Lista de nombres de blobs guardados en bucket respectivo
            normalization (string): Indica si se debe implementar la transformación
            de normalización para el clasificador multi-etiqueta ('mlc') o para el 
            detector de objetos ('ob')
            device (string): Indica qué dispositivo se quiere usar: 'cuda' o 'cpu'
        
        Returns:
            pro_img (tensor): Retorna la imagen procesada como tensor lista para la predicción.
        """
        
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        self.bucket = bucket
        self.blob_name_list = blob_name_list
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
        return len(self.blob_name_list)

    def __getitem__(self, idx):
        
        blob_name = self.blob_name_list[idx]
        imagename = blob_name.split('/')[1]
        blob = self.bucket.get_blob(blob_name)
        try:
            b = blob.download_as_bytes()
            image_cv = np.asarray(bytearray(b), dtype="uint8")
            foto = cv2.imdecode(image_cv, cv2.IMREAD_COLOR)
            area = foto.shape[0]*foto.shape[1]
            url = 'https://prdadessacorptrl.blob.core.windows.net/cl-images/' + imagename
            # se considera negro pixeles menores o iguales a 30
            norm_img0 = foto/30
            norm_img1 = np.where(norm_img0 <= 1 , 0, 1)
            number_of_black_pix = np.sum(norm_img1 == 0)
            size_img = area*np.shape(foto)[2]
            ratio = number_of_black_pix/size_img
            error_lectura = 0
        except:
            foto = np.zeros((480,480,3), np.uint8)
            area = foto.shape[0]*foto.shape[1]
            url = 'https://prdadessacorptrl.blob.core.windows.net/cl-images/' + imagename
            ratio = 1
            error_lectura = 1
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
        return pro_img, area, url, ratio, error_lectura

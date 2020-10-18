from torch.utils.data import Dataset
import cv2
import os 
import torch
import numpy as np


class CelebData(Dataset):
    
    def  __init__(self, root_dir):
        super().__init__()

        self.upper_lips = []
        self.lower_lips = []
        self.left_eyes = []
        self.right_eyes = []
        self.images = []

        
        for folders in ['0','1'] :

            img_path = root_dir + '/CelebAMask-HQ-mask-anno/'  + folders

            if folders != '.DS_Store':
                if folders == '0':
                    snoi = '00000'
                    noi = '00000'
                for image in os.listdir(img_path):
                    if image != '.DS_Store':                        
                        if int(image[0:5]) > int(noi):
                            noi = image[0:5]
                for i in range(int(snoi), int(noi)+1):
                    img_no = format(i, '05d')                   
                    
                    flag = True
                    for suffix in ['_l_eye.png', '_r_eye.png', '_u_lip.png', '_l_lip.png']:
                        if os.path.isfile(img_path + '/' + img_no + suffix):
                            pass
                        else:
                            flag = False 

                    if flag is True:                             
                        self.left_eyes.append(img_path +'/' + img_no + '_l_eye.png' )                     
                        self.right_eyes.append(img_path +'/' + img_no + '_r_eye.png')                        
                        self.upper_lips.append(img_path +'/' + img_no + '_u_lip.png')                        
                        self.lower_lips.append(img_path + folders +'/' + img_no + '_l_lip.png')
                        self.images.append(root_dir + '/CelebA-HQ-img/' + str(int(img_no)) + '.jpg' )
                            
                    snoi = noi
    
    def __len__(self):

        return len(self.upper_lips)
    
    def preprocess_masks(self,idx):
        
        ulip = cv2.imread(self.upper_lips[idx])
        llip = cv2.imread( self.lower_lips[idx])
        leye = cv2.imread(self.left_eyes[idx])
        reye = cv2.imread(self.right_eyes[idx])

        ulip = cv2.cvtColor(ulip, cv2.COLOR_BGR2GRAY)
        llip = cv2.cvtColor(llip, cv2.COLOR_BGR2GRAY)
        leye = cv2.cvtColor(leye, cv2.COLOR_BGR2GRAY)
        reye = cv2.cvtColor(reye, cv2.COLOR_BGR2GRAY)

        mask = np.zeros((512,512))

        mask[ulip == 255] = 1
        mask[llip == 255] = 2
        mask[leye == 255] = 3
        mask[reye == 255] =   4

        return mask         

    
    def __getitem__(self,idx):

        img = cv2.imread(self.images[idx])
        img = cv2.resize(img, (512,512))
        mask  = self.preprocess_masks(idx)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
       
        img = torch.tensor(img, dtype=torch.float32).view(-1,512,512)
        mask = torch.tensor(mask, dtype = torch.long)
        return {
            'img': img,
            'masks': mask
        }




        
        


import pandas as pd
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    elif mode == "submit":
        img = pd.read_csv('submit_img.csv')
        return np.squeeze(img.values), None
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class CVLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.size = 224
        self.tansforms_resize = T.Compose([
            T.Resize((self.size, self.size)),
            ])
        self.tansforms_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def get_image(self, path):
        from PIL import Image 
        if self.mode == 'train':
            self.size = 224
            self.tansforms_resize = T.Compose([
                T.Resize((self.size, self.size)),
                ])
            return Image.open(path).convert('RGB')
 
        else:
            return Image.open(path).convert('RGB')
    
    def img1to3(self, img):
        img = np.expand_dims(img, axis=2)
        a = img.copy()
        temp = np.concatenate((a,a),axis=2)
        a = np.concatenate((temp,a),axis=2)
        
        return a
    
    def data_aug(self, img):
        import random
        
        ##########################  Rotation  ##########################
        rand_int = random.randint(0,99)
        
        if rand_int < 25:
            img = np.rot90(img, 1)
        elif rand_int >= 25 and rand_int < 50:
            img = np.rot90(img, 2)
        elif rand_int >= 50 and rand_int < 75:
            img = np.rot90(img, 3)

        ##########################  Rotation  ##########################

        return img

    def __getitem__(self, index):
        """something you should implement here"""
        
        sample_id = self.img_name[index]
        img_path = "./data/" + "%06d.jpg" % sample_id 
        img = self.get_image(img_path)
        img = self.tansforms_resize(img)
        
        img = np.asarray(img)
        # if self.mode == 'train':
        #     img = self.data_aug(img)

        img = self.tansforms_normalize(img.copy())
        img = img.numpy()

        if self.mode == 'submit':
            return img, sample_id
        else:
            label = self.label[index]
            return img, label

if __name__ == '__main__':
    data_train = CVLoader("./","train") 
    train_loader = DataLoader(data_train, batch_size=2, shuffle=True, num_workers=4)
    for cur_it, (batch_data, batch_label) in enumerate(train_loader):
        print(cur_it)
    # for idx in range( len(data_train)):
    #     data = data_train[idx]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
#import torchvision
import os 
import numpy as np
import cv2
from matplotlib import pyplot as plt

from model import Compressor



class ImageData(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir


    def __getitem__(self, idx):
        self.img = cv2.imread(self.root_dir+'/'+str(idx)+'.png')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = self.img.transpose(2,0,1)
        self.img = self.normalization(self.img)
        self.img = self.to_tensor(self.img)
        return self.img
    
    def __len__(self):
        self.len = len(os.listdir(self.root_dir))
        return self.len
    
    def to_tensor(self, data):
        #self._toTensor = transforms.ToTensor()
        #data = self.toTensor(data)
        data = torch.from_numpy(data)
        return data
    
    def normalization(self, data):
        data = (data/255.0).astype(np.float32)
        return data

class TestData(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir


    def __getitem__(self, idx):
        self.img = cv2.imread(self.root_dir+'/'+str(idx+100)+'.png')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = self.img.transpose(2,0,1)
        self.img = self.normalization(self.img)
        self.img = self.to_tensor(self.img)
        return self.img
    
    def __len__(self):
        self.len = len(os.listdir(self.root_dir))
        return self.len
    
    def to_tensor(self, data):
        #self._toTensor = transforms.ToTensor()
        #data = self.toTensor(data)
        data = torch.from_numpy(data)
        return data
    
    def normalization(self, data):
        data = (data/255.0).astype(np.float32)
        return data

def train():
    model = Compressor()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using ', device)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

    train_dataset = ImageData(train_dir)
    train_loader = DataLoader(dataset = train_dataset,
                        batch_size = batch_size,
                        shuffle = True,
                        num_workers = 4)
    

    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, img in enumerate(train_loader):
            #print('epoch: ', epoch, ';\t iter: ', i, '\t data: ', img.shape)
            img = img.to(device)
            optimizer.zero_grad()
            
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % iter_per_epoch == iter_per_epoch-1:
                print('[epoch: %d, iter: %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / iter_per_epoch))
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    torch.save(model.state_dict(), save_dir+'/model.pth')
    



def evaluation(showcase_id=100):
    model = Compressor()
    model.load_state_dict(torch.load(save_dir+'/model.pth'))

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    test_dataset = TestData(test_dir)
    test_loader = DataLoader(dataset = test_dataset,
                        batch_size = 1,     #because of computation of SSIM, must be 1
                        shuffle = False,
                        num_workers = 1)
    
    total_ssim = 0.0
    with torch.no_grad():
        for img in test_loader:
            #img = img.to(device)
            rec_img = model(img)
            rec_img = denormalization(rec_img.numpy())
            img_hwc = img.numpy().squeeze(0).transpose(1,2,0)
            total_ssim += SSIM(rec_img, img_hwc)
    avg_ssim = total_ssim/num_testdata
    print('avg ssim on test dataset: ', avg_ssim)

    show_case(model, showcase_id)

    return avg_ssim
            
    


def show_case(model, test_id):
    imgdata = ImageData(test_dir)
    img = imgdata[test_id]
    img_4d = img.unsqueeze(0)
    rec_img = model(img_4d)
    rec_img = rec_img.detach().numpy()
    rec_img = denormalization(rec_img)
    img_hwc = img.numpy().transpose(1,2,0)
    
    ssim = SSIM(rec_img, img_hwc)
    print('showcase ssim: ', ssim)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_hwc)
    plt.subplot(1,2,2)
    plt.imshow(rec_img)
    plt.savefig('showcase.png')
    plt.show()
    plt.close()




def SSIM(img1, img2): 
    '''calculate SSIM
    img1, img2: [0, 255]
    img.ndim=[h,w,c]
    '''
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def denormalization(img):
    '''
    img:    4-d numpy array
    '''
    img[img>1.0] = 1.0
    img = (img*255.0).astype(np.uint8)
    img = img.transpose(0, 2, 3, 1)
    img = np.squeeze(img, axis=0)
    return img



if __name__ == '__main__':
    lr = 0.01
    lr_step = 10
    num_epoch = 100
    batch_size = 20
    num_traindata = 100
    num_testdata = 100
    iter_per_epoch = int((num_traindata+batch_size-1)/batch_size)
    max_iter = int((num_traindata+batch_size-1)/batch_size*num_epoch)
    train_dir = './train'
    test_dir = './test'
    save_dir = './model'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t', action='store_true', default=False)
    parser.add_argument('--showcase_id', '-i', default=100)
    args = parser.parse_args()
    if not args.test:
        train()
    else:
        showcase_id = int(args.showcase_id)
        evaluation(showcase_id)


    



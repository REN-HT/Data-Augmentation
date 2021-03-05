import torch
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
import numpy as np

class DataAugmentation(object):
    def __init__(self):
        super(DataAugmentation,self).__init__()

    def resize(self,img,boxes,size):
        #---------------------------------------------------------
        # 类型为 img=Image.open(path)，boxes:Tensor，size:int
        # 功能为：将图像长和宽缩放到指定值size，并且相应调整boxes
        #---------------------------------------------------------
        w,h=img.size
        sw=size/w
        sh=size/h
        return img.resize((size, size), Image.BILINEAR), boxes*torch.Tensor([sw, sh, sw, sh])

    def resize_(self, img, boxes, size):
        #-----------------------------------------------------------
        # 类型为 img=Image.open(path)，boxes:Tensor，size:int
        # 功能为：将图像短边缩放到指定值size,保持原有比例不变，并且相应调整boxes
        #-----------------------------------------------------------
        w,h=img.size
        min_size=min(w,h)
        sw=sh=size/min_size
        ow=int(sw*w+0.5)
        oh=int(sh*h+0.5)
        return img.resize((ow, oh), Image.BILINEAR), boxes*torch.Tensor([sw,sh,sw,sh])

    def random_flip_horizon(self, img, boxes):
        #-------------------------------------
        # 随机水平翻转
        #-------------------------------------
        if np.random.random() > 0.5:
            transform=transforms.RandomHorizontalFlip()
            img=transform(img)
            w = img.width
            xmin=w-boxes[:,2]
            xmax=w-boxes[:,0]
            boxes[:,0]=xmin
            boxes[:,2]=xmax
        return img, boxes

    def random_flip_vertical(self, img, boxes):
        #-------------------------------------
        # 随机垂直翻转
        #-------------------------------------
        if np.random.random()>0.5:
            transform=transforms.RandomVerticalFlip()
            img=transform(img)
            h=img.height
            ymin=h-boxes[:, 3]
            ymax=h-boxes[:, 1]
            boxes[:, 1]=ymin
            boxes[:, 3]=ymax
        return img, boxes

    def center_crop(self, img, boxes):
        #-------------------------------------
        # 中心裁剪
        #-------------------------------------
        w,h=img.size
        size=min(w,h)
        transform=transforms.CenterCrop(size)
        img=transform(img)
        sw,sh=size/w, size/h
        boxes=boxes*torch.Tensor([sw, sh, sw, sh])
        boxes=boxes.clamp(min=0, max=size)
        return img, boxes

    #------------------------------------------------------
    # 以下img皆为Tensor类型
    #------------------------------------------------------

    def random_bright(self, img, u=32):
        #-------------------------------------
        # 随机亮度变换
        #-------------------------------------
        if np.random.random()>0.5:
            alpha=np.random.uniform(-u, u)/255
            img+=alpha
            img=img.clamp(min=0.0, max=1.0)
        return img

    def random_contrast(self, img, lower=0.5, upper=1.5):
        #-------------------------------------
        # 随机增强对比度
        #-------------------------------------
        if np.random.random()>0.5:
            alpha=np.random.uniform(lower, upper)
            img*=alpha
            img=img.clamp(min=0, max=1.0)
        return img

    def random_saturation(self, img,lower=0.5, upper=1.5):
        #-----------------------------------------------
        # 随机饱和度变换，针对彩色三通道图像，中间通道乘以一个值
        #-----------------------------------------------
        if np.random.random()>0.5:
            alpha=np.random.uniform(lower, upper)
            img[1]=img[1]*alpha
            img[1]=img[1].clamp(min=0,max=1.0)
        return img

    def add_gasuss_noise(self, img, mean=0, std=0.1):
        noise=torch.normal(mean,std,img.shape)
        img+=noise
        img=img.clamp(min=0, max=1.0)
        return img

    def add_salt_noise(self, img):
        noise=torch.rand(img.shape)
        alpha=np.random.random()
        img[noise[:,:,:]>alpha]=1.0
        return img

    def add_pepper_noise(self, img):
        noise=torch.rand(img.shape)
        alpha=np.random.random()
        img[noise[:,:,:]>alpha]=0
        return img

    def mixup(self, img1, img2, box1, box2,alpha=2):
        weight=np.random.beta(alpha, alpha)
        print(weight)
        miximg=weight*img1+(1-weight)*img2
        if weight > 0.5:
            return miximg, box1
        return miximg, box2


    def draw_img(self, img, boxes):
        draw=ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='yellow', width=2)
        img.show()


D=DataAugmentation()
img1=Image.open('C:/AllProgram/testimage/superResolution/bird_GT.bmp')
img2=Image.open('C:/AllProgram/testimage/superResolution/lenna.bmp')

box1=torch.Tensor([[92,20,261,190]])
box2=torch.Tensor([[23,18,97,123],
                    [45,65,194,225]])

to_tensor=transforms.ToTensor()
to_image=transforms.ToPILImage()

img1,box1=D.resize(img1,box1,256)
img2,box2=D.resize(img2,box2,256)
img1=to_tensor(img1)
img2=to_tensor(img2)
miximg,box=D.mixup(img1,img2,box1,box2)

miximg=to_image(miximg)
D.draw_img(miximg,box)






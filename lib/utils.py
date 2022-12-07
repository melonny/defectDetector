import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms


def imshow(img,text=None,should_save=False):
    # transform = transforms.ToPILImage()
    # img = transform(img)
    #展示一幅tensor图像，输入是(C,H,W)
    npimg = img.numpy() #将tensor转为ndarray
    npimg = (npimg * 255).astype(np.uint8)
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #转换为(H,W,C)
    plt.show()
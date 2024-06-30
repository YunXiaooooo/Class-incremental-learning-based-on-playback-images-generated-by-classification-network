from PIL import Image
import math
import random
import numpy as np
import cv2
from torchvision import transforms

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """
    def __init__(self, snr, p=0.5):
        assert isinstance(snr, float) and (isinstance(p, float))  # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:  # 概率判断
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr  # 信噪比。信噪比0.9，说明信号占90%
            noise_pct = (1 - self.snr)  # 噪声占比0.1
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

class DislocationTranslationAndToTensor(object):
    """RG通道错位平移并转换为张量
    Args:
        MaxDistance （int）: x或y方向上错位最大距离，RG通道各占0.5*MaxDistance
        p (float): 概率值，依概率执行该操作
    """
    def __init__(self, MaxDistance=5, p=0.5):
        assert isinstance(MaxDistance, int) and (isinstance(p, float))
        self.MaxDistance = MaxDistance
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            tensor
        """
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow("bgr", image)
        B, G, R = cv2.split(image)
        R = cv2.copyMakeBorder(R, self.MaxDistance, self.MaxDistance, self.MaxDistance, self.MaxDistance,
                               cv2.BORDER_REPLICATE)
        G = cv2.copyMakeBorder(G, self.MaxDistance, self.MaxDistance, self.MaxDistance, self.MaxDistance,
                               cv2.BORDER_REPLICATE)
        # print("R shape", R.shape)
        # print("G shape", G.shape)
        # print("B shape", B.shape)
        # cv2.imshow("R", R)
        # cv2.imshow("G", G)
        # cv2.imshow("B", B)
        if random.uniform(0, 1) < self.p:  # 概率判断
            x_offset = random.randint(0 - self.MaxDistance//2, self.MaxDistance//2)
            y_offset = random.randint(0 - self.MaxDistance // 2, self.MaxDistance // 2)
            # print("x_offset=", x_offset, "y_offset=", y_offset)
            rows, cols = R.shape
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            R = cv2.warpAffine(R, M, (cols, rows))
            # cv2.imshow("Rb", R)

        if random.uniform(0, 1) < self.p:  # 概率判断
            x_offset = random.randint(0 - self.MaxDistance//2, self.MaxDistance//2)
            y_offset = random.randint(0 - self.MaxDistance // 2, self.MaxDistance // 2)
            # print("x_offset=", x_offset, "y_offset=", y_offset)
            rows, cols = G.shape
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            G = cv2.warpAffine(G, M, (cols, rows))
            # cv2.imshow("Gb", G)

        size = R.shape
        h = size[0]
        w = size[1]
        R = R[self.MaxDistance:h - self.MaxDistance, self.MaxDistance:w - self.MaxDistance]
        transform_temp = transforms.Compose([transforms.ToTensor()])
        R_tensor = transform_temp(R)
        # print(R_tensor.size())
        # print(R_tensor)
        size = G.shape
        h = size[0]
        w = size[1]
        G = G[self.MaxDistance:h - self.MaxDistance, self.MaxDistance:w - self.MaxDistance]
        transform_temp = transforms.Compose([transforms.ToTensor()])
        G_tensor = transform_temp(G)
        # print(G_tensor.size())
        # print(G_tensor)
        rg_tensor = torch.cat((R_tensor, G_tensor), 0)
        # print(rg_tensor.size())
        # cv2.waitKey(0)

        return rg_tensor

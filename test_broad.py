import cv2
import numpy as np
base = '/data/lyx/dataset/small90/small90/bike2/'
gt = np.loadtxt(base + 'groundtruth_rect.txt', delimiter=',')
for i in [90, 184]:
    img1 = cv2.imread(base + f'img/{i:06d}.jpg')
    img2 = img1.copy()
    b1 = gt[i-1].astype(int)
    b3 = gt[(i-1)*3].astype(int)
    cv2.rectangle(img1, (b1[0], b1[1]), (b1[0]+b1[2], b1[1]+b1[3]), (0, 255, 0), 2)
    cv2.rectangle(img2, (b3[0], b3[1]), (b3[0]+b3[2], b3[1]+b3[3]), (0, 0, 255), 2)
    cv2.imwrite(f'debug_bike_{i}_seq.jpg', img1)
    cv2.imwrite(f'debug_bike_{i}_mul3.jpg', img2)

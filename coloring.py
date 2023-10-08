import numpy as np
import torch
from libtiff import TIFF
import cv2
from torch.utils.data import DataLoader
import os
from mydata import MyData, MyData1
from tqdm import tqdm


TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ms4_tif = TIFF.open('../Remote Data/image/ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()

pan_tif = TIFF.open('../Remote Data/image/pan.tif', mode='r')
pan_np = pan_tif.read_image()

label_np = np.load("../Remote Data/image6/label.npy")


Ms4_patch_size = 16
Interpolation = cv2.BORDER_REFLECT_101

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)

Pan_patch_size = Ms4_patch_size * 4
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)

# label_np=label_np.astype(np.uint8)
label_np = label_np - 1

label_element, element_count = np.unique(label_np, return_counts=True)
Categories_Number = len(label_element) - 1
label_row, label_column = np.shape(label_np)
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy = np.array([[]] * Categories_Number).tolist()
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)

count = 0
for row in range(label_row):
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])

for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)

    ground_xy[categories] = ground_xy[categories][shuffle_array]

shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]
ground_xy_all_label = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    for i in range(categories_number):
        ground_xy_all_label.append(ground_xy[categories][i])

ground_xy_all_label = np.array(ground_xy_all_label)
ground_xy_all_label = torch.from_numpy(ground_xy_all_label).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)
ms4 = np.array(ms4).transpose((2, 0, 1))

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)


all_label_data = MyData1(ms4, pan, ground_xy_all_label, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
all_label_data_loader = DataLoader(dataset=all_label_data, batch_size=TEST_BATCH_SIZE,shuffle=False,num_workers=8, drop_last=True)
all_data_loader = DataLoader(dataset=all_data, batch_size=TEST_BATCH_SIZE,shuffle=False,num_workers=8, drop_last=True)

cnn = torch.load('model.pkl')
cnn.cuda()

class_count = np.zeros(Categories_Number)
out_clour = np.zeros((label_row, label_column, 3))
def clour_model(cnn, data, filename):
    loop = tqdm(data, desc='Test', ncols=130)
    for ms4, pan, gt_xy in loop:
        ms4 = ms4.cuda()
        pan = pan.cuda()
        with torch.no_grad():
            output = cnn(ms4, pan)
        pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()
        gt_xy = gt_xy.numpy()
        for k in range(len(gt_xy)):
            if pred_y_numpy[k] == 0:
                class_count[0] = class_count[0] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
            elif pred_y_numpy[k] == 1:
                class_count[1] = class_count[1] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
            elif pred_y_numpy[k] == 2:
                class_count[2] = class_count[2] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [33, 145, 237]
            elif pred_y_numpy[k] == 3:
                class_count[3] = class_count[3] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
            elif pred_y_numpy[k] == 4:
                class_count[4] = class_count[4] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [240, 32, 160]
            elif pred_y_numpy[k] == 5:
                class_count[5] = class_count[5] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [221, 160, 221]
            elif pred_y_numpy[k] == 6:
                class_count[6] = class_count[6] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [140, 230, 240]
            elif pred_y_numpy[k] == 7:
                class_count[7] = class_count[7] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 255]
            elif pred_y_numpy[k] == 8:
                class_count[8] = class_count[8] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
            elif pred_y_numpy[k] == 9:
                class_count[9] = class_count[9] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [127, 255, 0]
            elif pred_y_numpy[k] == 10:
                class_count[10] = class_count[10] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 255]

    print(class_count)
    cv2.imwrite(filename+".png", out_clour)

clour_model(cnn,  all_label_data_loader, 'all_label_data_1')
clour_model(cnn,  all_data_loader, 'all_data_1')
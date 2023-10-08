import numpy as np
import torch
from libtiff import TIFF
import cv2
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import torch.optim as optim
from mydata import MyData, MyData1
from model import generate_model
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


EPOCH = 30
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.001
Train_Rate = 0.02
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



ms4_tif = TIFF.open('../Remote Data/image/ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()

pan_tif = TIFF.open('../Remote Data/image/pan.tif', mode='r')
pan_np = pan_tif.read_image()

label_np = np.load("../Remote Data/image/label.npy")




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

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    for i in range(categories_number):
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
        else:
            ground_xy_test.append(ground_xy[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
    label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

print('num_train_sample：', len(label_train))
print('num_test_sample：', len(label_test))

ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)
ms4 = np.array(ms4).transpose((2, 0, 1))

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)

train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
all_data_loader = DataLoader(dataset=all_data, batch_size=TEST_BATCH_SIZE,shuffle=False,num_workers=0)


model = generate_model(18).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0.0
    loop = tqdm(train_loader, desc='TRAIN EPOCH:{}'.format(epoch), ncols=130)
    for step, (ms, pan, label, _) in enumerate(loop):
        ms, pan, label = ms.to(device), pan.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(ms, pan)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        loss = F.cross_entropy(output, label.long())
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))

def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    l = 0
    print('testing...')
    with torch.no_grad():
        loop = tqdm(test_loader, desc='Test', ncols=130)
        for data, data1, target, _ in loop:
            l += 1
            data, data1, target = data.to(device), data1.to(device), target.to(device)
            output = model(data, data1)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
            if l == 1:
                y_pred = pred.cpu().numpy()
                y_true = target.cpu().numpy()
            else:
                y_pred = np.concatenate((y_pred, pred.cpu().numpy()), axis=0)
                y_true = np.concatenate((y_true, target.cpu().numpy()), axis=0)
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))
        con_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print("confusion_matrix", con_mat)

        # 计算性能参数
        all_acr = 0
        p = 0
        column = np.sum(con_mat, axis=0)
        line = np.sum(con_mat, axis=1)
        for i, clas in enumerate(con_mat):
            precise = clas[i]
            all_acr = precise + all_acr
            acr = precise / column[i]
            recall = precise / line[i]
            f1 = 2 * acr * recall / (acr + recall)
            temp = column[i] * line[i]
            p = p + temp
            print("Category %d: || PRECISION: %.7f || RECALL: %.7f || F1: %.7f " % (i, acr, recall, f1))
        OA = np.trace(con_mat) / np.sum(con_mat)
        print('OA:', OA)

        AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))
        print('AA:', AA)

        Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
        Kappa = (OA - Pc) / (1 - Pc)
        print('Kappa:', Kappa)
        torch.save(model, 'model.pkl')

for epoch in range(1, EPOCH+1):
    train_model(model,  train_loader, optimizer, epoch)

test_model(model, test_loader)

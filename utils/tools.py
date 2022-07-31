import random

import joblib
import numpy as np
import torch.utils.data as util_data
from torch.utils import data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json

num = 0#os.cpu_count()

def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m","nuswide_21_joblib"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20
    elif config["dataset"] == "deepfashion":
        config["topK"] = 1000
        config["n_class"] = 14

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "./dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "./dataset/COCO_2014/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "./dataset/"

    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)




def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)] #类别转换，target是2，那么转成onehot就是[0,1,0,0,0,0,0,0,0,0]
        return img, target, index


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformTrain = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = MyCIFAR10(root='dataset/',
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root='dataset/',
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root='dataset/',
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    # 修改一下train_dataset的增强器
    train_dataset.transform = transformTrain
    train_dataset.transforms = transformTrain

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


class nusImageList(data.Dataset):
    def __init__(self, data_path, transform):
        entry = joblib.load(data_path)
        #images=np.array(entry["images"])
        images = entry["images"]
        self.imgs = []
        for img in images:
            image = Image.fromarray(img)
            self.imgs.append(image)

        self.targets = entry["targets"]
        self.transform = transform
        self.targets = torch.LongTensor(self.targets)

    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.transform(img)
        target = self.targets[index]
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def nus_wide_init(config):
    config["topK"] = 5000
    config["n_class"] = 21
    train_path = "dataset/nus-wide-joblib/train_dataset_64"
    test_path = "dataset/nus-wide-joblib/test_dataset_64"
    dababase_path = "dataset/nus-wide-joblib/database_dataset_64"

    train_data = nusImageList(train_path,transform=image_transform(config["resize_size"], config["crop_size"], "train_set"))
    test_data = nusImageList(test_path,transform=image_transform(config["resize_size"], config["crop_size"], "test_set"))
    database_data = nusImageList(dababase_path,transform=image_transform(config["resize_size"], config["crop_size"], "database_set"))

    print("train_data:", len(train_data))
    print("test_data:", len(test_data))
    print("database_data:", len(database_data))

    train_data = util_data.DataLoader(train_data,batch_size=config["batch_size"],shuffle=True, num_workers=num)
    test_data = util_data.DataLoader(test_data,batch_size=config["batch_size"],shuffle=True, num_workers=num)
    database_data = util_data.DataLoader(database_data,batch_size=config["batch_size"],shuffle=True, num_workers=num)

    return train_data, test_data, database_data, len(train_data), len(test_data), len(database_data)

def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    if "nuswide_21_joblib" in config["dataset"]:
        return nus_wide_init(config)

    if "deepfashion" in config["dataset"]:
        return deepfashion_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=num)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def deepfashion_dataset(config):
    path = "dataset/img_highres/WOMEN"
    test_data_list,train_data_list,database_list = getDeepFashionFile(path)

    train_data = DeepFashionImageList(train_data_list,transform=image_transform(config["resize_size"], config["crop_size"], "train_set"),class_count=config["n_class"])
    test_data = DeepFashionImageList(test_data_list,transform=image_transform(config["resize_size"], config["crop_size"], "test_set"),class_count=config["n_class"])
    database_data = DeepFashionImageList(database_list,transform=image_transform(config["resize_size"], config["crop_size"], "database_set"),class_count=config["n_class"])

    print("train_data:", len(train_data))
    print("test_data:", len(test_data))
    print("database_data:", len(database_data))

    train_data = util_data.DataLoader(train_data,batch_size=config["batch_size"],shuffle=True, num_workers=num)
    test_data = util_data.DataLoader(test_data,batch_size=config["batch_size"],shuffle=True, num_workers=num)
    database_data = util_data.DataLoader(database_data,batch_size=config["batch_size"],shuffle=True, num_workers=num)

    return train_data, test_data, database_data, len(train_data), len(test_data), len(database_data)


class DeepFashionImageList(object):

    def __init__(self, data_path, transform,class_count = 14):
        self.imgs = []
        for file_class in  data_path:
            self.imgs.extend(file_class)
        self.targets = []
        for i,file_class2 in enumerate(data_path):
            label = np.ones(len(file_class2)) - 1 + i
            onthot = torch.eye(class_count)[label, :]
            self.targets.extend(onthot.numpy()) #torch.LongTensor(self.targets)

        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.targets[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

import glob
import os
def getDeepFashionFile(path):
    file_list = [] #二级文件目录，其中
    for class_list in os.listdir(path):
        class_file_name = []
        for sub_class_list in os.listdir(os.path.join(path,class_list)):
            last_list = glob.glob(os.path.join(path, class_list, sub_class_list, '*.jpg'))
            class_file_name.extend(last_list)
        file_list.append(class_file_name)

    train_data_list = []
    test_data_list = []
    database_list = file_list#[]
    test_num = 100
    train_num = 300

    for a in file_list:
        test_data_list.append(a[:test_num])
        end_num = test_num+train_num
        if end_num>len(a):
            end_num = len(a)
        if len(a[test_num:])<train_num:
            begin_num = len(a)-train_num
        else:
            begin_num = test_num
        train_data_list.append(a[begin_num:end_num])
        #database_list.append(a[test_num:])

    return test_data_list,train_data_list,database_list







def compute_result2(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs),torch.cat(bs).sign(), torch.cat(clses)




def compute_result_fc(dataloader, net,fc, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        u=net(img.to(device))
        u= fc(u)
        bs.append((u).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def CalcTopMap2(database_code,
                query_code,
                database_labels,
                query_labels,
                topk=5000,
                device=None,
                ):
    """
    Fix the Tie Problem 。
    Calculate mean average precision(map).

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = torch.zeros(1).to(device)
    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance

        hamming_index = torch.argsort(hamming_dist)
        retrieval = retrieval[hamming_index][:topk]
        #
        hamming_order = hamming_dist[hamming_index][:topk]

        k = 0
        retrieval = retrieval.cpu().numpy()
        hamming_order = hamming_order.cpu().numpy()
        last_dis = hamming_order[0]

        #         start=time.time()
        for j in range(0, topk):
            if last_dis != hamming_order[j]:
                k = j
                last_dis = hamming_order[j]

            if retrieval[j] == 1 and k != j:
                retrieval[k] = 1
                k += 1
                retrieval[j] = 0
        #         print(time.time()-start)
        retrieval = torch.from_numpy(retrieval).to(device)
        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return float(mean_AP)


def random_seed(seed=np.random.randint(1,10000)):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)



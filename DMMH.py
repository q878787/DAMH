from datetime import datetime

from loguru import logger

from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
import math

torch.multiprocessing.set_sharing_strategy('file_system')


# DTSH(ACCV2016)
# paper [Deep Supervised Hashing with Triplet Labels](https://arxiv.org/abs/1612.03900)
# code [DTSH](https://github.com/Minione/DTSH)

def get_config():
    config = {
        "alpha": 5,
        "lambda": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DMMH6]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        #"dataset": "cifar10",
        #"dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        #"dataset": "imagenet",
        #"dataset": "nuswide_21",
        #"dataset" : "nuswide_21_joblib",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 1000,
        "test_map": 10,
        "save_path": "save/DMMH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class DMMHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DMMHLoss, self).__init__()
        self.bit = bit

    def forward(self, u, y, ind, config):

        inner_product = u @ u.t()
        s = y @ y.t() > 0
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                k=self.bit
                pi = math.pi
                t1 = (theta_positive-(-k))/(2*k) * pi/2
                t2 = (theta_negative-(-k))/(2*k) * pi/2
                triple = 6*(torch.cos(t2.unsqueeze(1))-torch.cos(t1.unsqueeze(0)))
                #triple = torch.cos(theta_negative.unsqueeze(1))-torch.cos(theta_positive.unsqueeze(0))
                #triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - config["alpha"]).clamp(min=-100,max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()


        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = config["lambda"] * (u - u.sign()).pow(2).mean()

        return loss1 + loss2


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DMMHLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        logger.info(f"epoch:{epoch} train_loss:{train_loss}")
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)
            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            logger.info(f"epoch:{epoch} MAP:{mAP}")
            if mAP > Best_mAP:
                Best_mAP = mAP
                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    # torch.save(net.state_dict(),
                    #           os.path.join(config["save_path"], config["info"] + "-" + str(mAP) + "-model.pt"))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)

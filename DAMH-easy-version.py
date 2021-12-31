from utils.tools import *
#from utils.tool2 import *

from network import *
import os
import torch
import torch.optim as optim
import time
import numpy as np
from loguru import logger

torch.multiprocessing.set_sharing_strategy('file_system')
# you need to set this config information to train DAMH
def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 1e-5}},
        "info": "[DAMH_low0_upperbd2_lrbd8_q01_b48_AlexNet_Adam_cifar-2]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 256,
        "net": AlexNet,
        # "net":ResNet,
        "dataset": "cifar10",
        # "dataset": "cifar10-2",
        #"dataset": "imagenet",
        #"dataset": "nuswide_21",
        #"dataset":"nuswide_21_joblib"
        "epoch": 29,
        "test_map": 7,
        "save_path": "save/[DAMH_low0_upperbd2_lrbd8_q01_b48_AlexNet_Adam_cifar-2]",
        "device": torch.device("cuda:0"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class DAMHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DAMHLoss, self).__init__()
        self.y_p = 0.5 #Centrosymmetric point
        self.right = bit / 8  
        self.left = self.right / 2
        self.lowerBound = 0
        self.upperBound = bit / 2
        self.percent = 9 / 10

    def forward(self, u, y, ind, config):
        u = u.tanh()
        s = y @ y.t() > 0
        inner = u @ u.t()

        posL = 0
        navL = 0
        count = 0
        for row in range(u.shape[0]):
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                #getting dissimilar-pair and similar-pair
                similar = inner[row][s[row] == 1]
                dissimilar = inner[row][s[row] == 0]
                #sorting
                similar_temp, idx = torch.sort(similar, descending=True)
                dissimilar_temp, idx2 = torch.sort(dissimilar)

                #DAMH_similar
                meanS = torch.mean(similar).clamp(min=self.lowerBound, max=self.upperBound).item()
                # percent can reduce interference of dissimilarMaxInner
                dissimilarMaxInner = dissimilar_temp[int(len(dissimilar_temp) * self.percent):].mean().clamp(min=self.lowerBound,max=self.upperBound).item()
                #getting xp
                BP = meanS - (self.upperBound - meanS) / self.upperBound * np.abs((meanS - dissimilarMaxInner))
                #getting hard or easy samples
                similar_easy = similar[similar > BP]
                similar_hard = similar[similar < BP]
                # calc
                a, c, d, g = self.calcParameter(BP, self.y_p, self.left, self.right)
                f_similar_easy = c * similar_easy + d
                f_similar_hard = a * c * similar_hard + g
                similar_easy_loss = self.DPSHLoss(True, f_similar_easy)
                similar_hard_loss = self.DPSHLoss(True, f_similar_hard)

                # DAMH_dissimilar
                meanDS = torch.mean(dissimilar).clamp(min=self.lowerBound, max=self.upperBound).item()
                similarMinInner = similar_temp[int(len(similar_temp) * self.percent):].mean().clamp(min=self.lowerBound, max=self.upperBound).item()
                BP_ds = meanDS + meanDS / self.upperBound * np.abs((meanDS - similarMinInner))
                dissimilar_easy = dissimilar[dissimilar < BP_ds]
                dissimilar_hard = dissimilar[dissimilar > BP_ds]
                a, c, d, g = self.calcParameter(BP_ds, self.y_p, self.left, self.right)
                f_dissimilar_easy = c * dissimilar_easy + d
                f_dissimilar_hard = a * c * dissimilar_hard + g
                dissimilar_easy_loss = self.DPSHLoss(False, f_dissimilar_easy)
                dissimilar_hard_loss = self.DPSHLoss(False, f_dissimilar_hard)

                #mean Loss
                similar_loss = torch.cat((similar_easy_loss, similar_hard_loss), dim=0)
                dissimilar_loss = torch.cat((dissimilar_easy_loss, dissimilar_hard_loss), dim=0)
                posL += similar_loss.mean()
                navL += dissimilar_loss.mean()

        if count != 0:
            posL = posL / count
            navL = navL / count
        else:
            posL = 0
            navL = 0

        Q_Loss = config["alpha"] * (u - u.sign()).pow(2).mean()

        return posL + navL + Q_Loss

    def DPSHLoss(self, s, fx):
        # s * fx + torch.log((1 + torch.exp(-fx)))
        if s == 1:
            losses = fx + torch.log((1 + torch.exp(-fx)))
        else:
            losses = torch.log((1 + torch.exp(-fx)))
        return losses

    #See the paper for details
    def calcParameter(self, BP, y_p, left, right):
        # y1 = 1/(1+e^(cx+d))
        # 1)c=1/right *log((y_p)/99*(1-y_p))
        c = 1 / right * np.log(y_p / (99 * (1 - y_p)))
        # 2)d=log((r-y_p)/y_p)- c*BP
        d = np.log((1 - y_p) / y_p) - c * BP
        # y2 = r/(1+e^(ax+g))
        # 1) a = -1/(left*c) *log( (99*y_p)/(1-y_p) )
        a = -1 / (left * c) * np.log((99 * y_p) / (1 - y_p))
        # 2)g = log((1-y_p)/y_p)-a*c*BP
        g = np.log((1 - y_p) / y_p) - a * c * BP

        return a, c, d, g


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = DAMHLoss(config, bit)

    logger.add("runs/log/" + config['info'] + ".log", level="INFO")
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

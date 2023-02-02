from copy import deepcopy
from pprint import pprint
from dataset import (AudioDataset,
                     get_a_batch_samples,
                     get_feature_shape)
from models.cnn import CNNNetwork
import torch, sys
from io import StringIO
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm, trange
from config import config
# from tool.output import enable_stdout, disable_stdout
from tool.draw import get_figure_BytesIO
from models.augment import Mixup
# from tool.telegram import send_configed_message



class Trainer():
    def __init__(self):
        self.trainloader = DataLoader(
                AudioDataset(train=True, test_fold_num=config["testfold"]),
                batch_size=config["batch_size"],
                shuffle=True)
        self.testloader = DataLoader(
                AudioDataset(train=False, test_fold_num=config["testfold"]),
                batch_size=config["batch_size"],
                shuffle=False)
        self.model = CNNNetwork().cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config["lr"]))
        self.metric = Accuracy(task="multiclass", num_classes=config["num_class"]).cuda()
        self.record_epoch = dict(train_loss=0, val_loss=0, train_accuracy=0, val_accuracy=0)
        self.record_all_epoch = []

    def show_init_message(self):
        pprint(config["preprocessing"]["mel_arg"])
        print("input feature shape", get_feature_shape())
        print("train datafolder", config["feature_folder"])
        self.model.summary()

    def train_a_batch(self):
        x, y = get_a_batch_samples()
        x = x.cuda(); y = y.cuda()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        print(loss)

    def train_a_epoch(self):
        self.model.train()
        self.metric.reset()
        total_loss = 0
        for x, y in self.trainloader:
            x = x.cuda(); y = y.cuda()
            if config["mixup"] is True:
                mixup = Mixup(config["batch_size"], .2)
                x = mixup.get_mixup_x(x)
                pred = self.model(x)
                loss = mixup.get_loss(pred,y)
            else:
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.metric.update(pred, y)
            total_loss += loss
        self.record_epoch["train_loss"] = total_loss.item()
        self.record_epoch["train_accuracy"] = self.metric.compute()

    def train_all_epoch(self, show=True, using_bar=False):
        print_file = sys.stdout if show is True else StringIO()  # 抑制輸出
        iteration = trange(config["num_epoch"]) \
            if using_bar else range(config["num_epoch"])
        for epoch in iteration:
            # if epoch == 30: breakpoint()
            self.train_a_epoch()
            self.validation()
            self.record_all_epoch.append(
                    deepcopy(dict(epoch=epoch, **self.record_epoch)))
            print(self.get_epoch_message(epoch), file=print_file)
        print(self.get_best_epoch_message(), file=print_file)

    def validation(self):
        self.model.eval()
        self.metric.reset()
        total_loss = 0
        for x, y in self.testloader:
            x = x.cuda(); y = y.cuda()
            pred = self.model(x)
            self.metric(pred, y)
            total_loss += self.loss_fn(pred, y)
        self.record_epoch["val_loss"] = total_loss.item()
        self.record_epoch["val_accuracy"] = self.metric.compute()

    def get_record_message(self, record):
        return (f"train/val   loss:{record['train_loss']:.3f} / {record['val_loss']:.3f}"
                f"  accuracy:{record['train_accuracy']:.3f} / {record['val_accuracy']:.3f}")

    def get_epoch_message(self, epoch):
        return f"epoch:{epoch} " + self.get_record_message(self.record_all_epoch[epoch])

    def get_best_epoch(self):
        return sorted(self.record_all_epoch,
                            key=lambda x: x["val_accuracy"],
                            reverse=True)[0]["epoch"]

    def get_best_epoch_message(self):
        return "best " + self.get_epoch_message(self.get_best_epoch())

    def get_fold_message(self, fold):
        return f"fold:{fold} " + self.get_best_epoch_message()

    def tqdm_bar(self, iterable):  # 拿掉 tqdm  123 it/s 的顯示
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} {remaining}{postfix}"
        return tqdm(iterable, bar_format=bar_format)

    def get_k_fold_message(self):
        acc = torch.zeros(5)  # assume always 5 fold
        for i, record in enumerate(self.fold_record_list):
            acc[i] = record["val_accuracy"]
        return f"k_fold mean:{acc.mean():.3f} std:{acc.std():.3f}"

    def train_k_fold(self, using_bar=False):
        self.fold_record_list = []
        for fold in range(5):
            config["test_fold_num"] = fold+1   # fold in [1,5]
            self.__init__()  # reset model, trainloader, testloader
            self.train_all_epoch(show=False, using_bar=using_bar)
            self.fold_record_list.append(
                    self.record_all_epoch[self.get_best_epoch()])
            print(self.get_fold_message(fold))
        print(self.get_k_fold_message())

    def get_draw_object(self, record_type = 'train_loss' ):  # loss, or val_accuracy
        if record_type not in ("val_accuracy", "train_loss"):
            raise RuntimeError("reocrd type must be 'val_accuracy' or 'train_loss'.")
        data = [ record[record_type] for record in self.record_all_epoch ]
        return get_figure_BytesIO( title=record_type, data = data )
        # for send messsage to telegram

if __name__ == "__main__":
    trainer = Trainer()
    trainer.show_init_message()
    # trainer.train_a_epoch()
    trainer.train_all_epoch()
    # send_configed_message(config, trainer.get_best_epoch_message())


# TODO

# trainer.train_k_fold()
# send_configed_message(config, trainer.get_k_fold_message())

    # save config.yaml to exp/dir

    # additional:
        # wandb logger
        # save image, train curve

    # preload model, only testing
    # only training
    # train all data

import sys  # add new path to search top level module
from models.cnn import CNNNetwork
from models.normalize import Normalize
from config import config

############################ TODO #######################

# 加入 augmentation 版本, 在model 上執行,不必修改訓練流程
class AugModel(CNNNetwork):
    def __init__(self):
        super().__init__()
        mean, std = self.get_train_data_info()
        self.normalize = Normalize(mean=mean, std=std)
    def forward(self, input_data):
        input_data = self.normalize( input_data)
        logit = super().forward(input_data)
        return logit
    def get_train_data_info(self):
        from config import config

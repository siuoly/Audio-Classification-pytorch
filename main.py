from trainer import Trainer
from tool.telegram import (send_configed_message,
                           send_telegram_photo,
                           send_telegram_photo)
from tool.draw import draw_image_from_BytesIO
from preprocessing import main as preprocessing
from config import config

def train_a_parameter(setting, keyword,value,fold=False,show=False,using_bar=False):
    setting[keyword] = value
    trainer = Trainer()
    # trainer.show_init_message()
    if not fold:
        trainer.train_all_epoch(show=show,using_bar=using_bar)
        print(trainer.get_best_epoch_message())
        send_configed_message(config, trainer.get_best_epoch_message() )
    else:
        trainer.train_k_fold(using_bar=using_bar)
        send_configed_message(config, trainer.get_k_fold_message() )


if __name__ == "__main__":
    preprocessing(show=True)
    trainer = Trainer()
    # trainer.train_all_epoch()


    config["num_epoch"] = 500
    # train_a_parameter(config,"lr",3e-4,fold=True,show=False,using_bar=True)  #mean.697 std:.
    train_a_parameter(config, "mixup", False,fold=False,show=True,using_bar=False)  #mean.717 std:.021
    # train_a_parameter(config, "mixup", True,fold=False,show=True,using_bar=False)  


    # config['lr'] = 8e-5
    # trainer = Trainer()
    # trainer.train_all_epoch(show=False,using_bar=True)
    # print(trainer.get_best_epoch_message())
    # send_configed_message(config, trainer.get_best_epoch_message() )


    # trainer.train_k_fold(using_bar=True)
    # print( trainer.get_k_fold_message() )
    # send_configed_message(config, trainer.get_k_fold_message() )

    # img = trainer.get_draw_object(record_type="loss")
    # send_telegram_photo( img )

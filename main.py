from trainer import Trainer
from tool.telegram import (send_configed_message,
                           send_telegram_photo)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_a_epoch()
    trainer.train_all_epoch()
    # trainer.get_k_fold_message()
    # tele

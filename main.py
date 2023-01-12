from trainer import Trainer
from tool.telegram import (send_configed_message,
                           send_telegram_photo,
                           send_telegram_photo)
from tool.draw import draw_image_from_BytesIO
from preprocessing import main as preprocessing
if __name__ == "__main__":
    preprocessing(show=False)

    trainer = Trainer()
    # trainer.train_a_epoch()
    trainer.train_all_epoch(show=True,using_bar=False)
    # trainer.train_k_fold(using_bar=False)

    # img = trainer.get_draw_object(record_type="loss")
    # send_telegram_photo( img )

    # trainer.get_k_fold_message()
    # tele

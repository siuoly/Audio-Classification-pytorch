#!/bin/python
import requests
from urllib import parse
import time
from tool.secret import token, chatID
import dataset

# e.g.
# send_telegram_msg("this is a good day")
def send_telegram_msg(msg: str,
                      token: str = token,
                      chatID: str = chatID):
    assert type(msg) == str, "傳入訊息必須為字串"
    url = f'https://api.telegram.org/bot{token}/sendMessage?chat_id={chatID}&text={msg}'
    requests.get(url)


def get_config_message(config):  # 2022/01/23 ./data/xxxdir/ batch=23
    # message = f"{time.ctime()}\n"
    message = config['dataset']['train_folder'] + "\n"
    message += config['message']
    return message


def send_configed_message(config, message):
    message = get_config_message(config) + '\n' + message
    send_telegram_msg(message)
    return message


# e.g.
# send_telegram_url_photo('https://i.imgur.com/R00cjSL.png')
def send_telegram_url_photo(photo_url ,
                            token: str = token,
                            chatID: str = chatID):
    photo_url = parse.quote(photo_url)  # 轉換url裡面的中文等符號
    send_text = 'https://api.telegram.org/bot' + token + '/sendPhoto?chat_id=' \
            + chatID + '&photo=' + photo_url
    print("send image url:", photo_url)
    response = requests.get(send_text)
    return response.json()



def send_telegram_photo(photo,
                        token: str = token,
                        chatID: str = chatID):
    url = 'https://api.telegram.org/bot' + token + '/sendPhoto'
    response = requests.post(url, data={'chat_id': chatID},
                             files={'photo': photo})
    photo.close()  # photo is a file object or steamIO
    return response.json()


# e.g. 1
# with open("example.png", 'rb') as iamge:
#     send_telegram_photo(image)

# e.g. 2
# from io import BytesIO
# res = requests.get("https://i.imgur.com/R00cjSL.png")
# image = BytesIO(res.content)
# send_telegram_photo(image)

#!/bin/python
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def get_figure_BytesIO(title, data):
    fig = plt.figure()  # 指定畫布,標題, 繪圖
    fig.suptitle(title)
    plt.plot( data ,figure=fig)
    buf = BytesIO()    # 把繪圖寫入 steam, 歸零 seek
    fig.savefig(buf, format='png')
    buf.seek(0)     # 為了後續開啟image正常
    plt.close(fig)  # 避免 後續plt.plot()重複出現前面標題
    return buf


def write_buffer_image_to_disk( buf, filename):
    with open( filename, 'wb') as f:
        f.write(buf.getvalue())

def draw_image_from_BytesIO(image):
    img = Image.open(image)
    plt.imshow(img)
    plt.show()


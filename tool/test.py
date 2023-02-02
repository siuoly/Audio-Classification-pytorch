
import io
import numpy as np
import sys

# with open('./example.png', 'rb') as f:
bio = io.BytesIO(np.arange(4,dtype=np.int8))
print(len(bio.getvalue()))
print(bio.getbuffer().nbytes)
print( bio.__sizeof__())
print( sys.getsizeof(bio))
print(bio.read(1))


def test_Mixup():
    from models.augment import Mixup
    import torch
    m = Mixup(3,2)
    print(m.random_indices, m.lamb)
    x = torch.arange(6).reshape(3,2,1)
    print(x)
    x = m(x)
    print(x)
    
test_Mixup()


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

    # print(f)

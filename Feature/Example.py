import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FaceDetection.Face as FD
from PIL import Image, ImageDraw




img = Image.open('./Images/data/farm1.staticflickr.com&2&2999787_a1cf5e862d_z.jpg')
croped = FD.crop_face(img)
for i in croped:
    i.resize((224,224)).show()
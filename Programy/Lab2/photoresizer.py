import PIL
import os
import os.path
from PIL import Image

f = r'E:\baza_ryjcow\fairface\fairface\val'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((50, 50))
    img.save(f_img)
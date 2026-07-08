import os
from PIL import Image

def create_icon(size):
    img = Image.new('RGB', (size, size), color = 'red')
    img.save(f'icons/icon{size}.png')

if not os.path.exists('icons'):
    os.makedirs('icons')

create_icon(16)
create_icon(48)
create_icon(128)
print("Icons created.")

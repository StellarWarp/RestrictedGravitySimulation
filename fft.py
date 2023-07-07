import taichi as ti
from PIL import Image
import numpy as np
from math import pi

ti.init(arch=ti.gpu)

size = 512

def read_image(path):
    image = Image.open(path).convert('LA')
    # resize to square image
    image = image.resize([size, size])
    data = np.asarray(image)
    data = np.cast[np.float32](data)
    # Convert to grey value within [0, 1]
    return data[:, :, 0] / data[:, :, 1]

def show_image(data, scale=True):
    if scale:
        data = data / np.max(data)
    data = data * 255.0
    c = np.ones_like(data) * 255.0
    data = np.stack([data, c], -1)
    data = np.cast[np.uint8](data)
    img = Image.fromarray(data, 'LA')
    img.show()

pixels = ti.var(dt=ti.f32, shape=(size, size))
results = ti.var(dt=ti.f32, shape=(size, size))

path = input('image path: ')
pixels.from_numpy(read_image(path))

@ti.kernel
def fourier():
    # parallel
    for k, l in results:
        v = ti.Vector([0.0, 0.0])
        for i in range(size):
            for j in range(size):
                center = size // 2
                kk = (k + center) % size
                ll = (l + center) % size
                angle = -2.0 * pi * (kk * i + ll * j) / float(size)
                p = ti.Vector([ti.cos(angle), ti.sin(angle)])
                v += pixels[i, j] * p
        center = size // 2
        results[k, l] = ti.log(1.0 + v.norm())

fourier()
data = results.to_numpy()
show_image(data)
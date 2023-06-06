import os
import numpy as np
import matplotlib.pyplot as plt

def get_sh(theta, phi):
    t = theta * np.pi / 180.
    p = phi * np.pi / 180.
    x = -np.sin(t) * np.cos(p)
    y = -np.cos(t) * np.cos(p)
    z = np.sin(p)
    # 1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-Y^2
    sh = [1, y, z, x, y * x, y * z, 3 * z * z - 1, x * z, x * x - y * y]
    return sh

def get_lighting(theta, phi, img_size=256):
    sh = get_sh(theta, phi)
    sh = np.squeeze(sh)
    shading = sh2lighting(sh, img_size=img_size)


    return shading, sh

def get_shading(normal, SH):
    sh_basis = SH_basis(normal)
    shading = np.matmul(sh_basis, SH)
    return shading

def SH_basis(normal):
    '''
       get SH basis based on normal
       normal is a Nx3 matrix
       return a Nx9 matrix
       The order of SH here is:
       1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    numElem = normal.shape[0]

    norm_X = normal[:, 0]
    norm_Y = normal[:, 1]
    norm_Z = normal[:, 2]

    sh_basis = np.zeros((numElem, 9))
    att = np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])
    sh_basis[:, 0] = 0.5 / np.sqrt(np.pi) * att[0]

    sh_basis[:, 1] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_Y * att[1]
    sh_basis[:, 2] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_Z * att[1]
    sh_basis[:, 3] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_X * att[1]

    sh_basis[:, 4] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_Y * norm_X * att[2]
    sh_basis[:, 5] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_Y * norm_Z * att[2]
    sh_basis[:, 6] = np.sqrt(5) / 4 / np.sqrt(np.pi) * (3 * norm_Z ** 2 - 1) * att[2]
    sh_basis[:, 7] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_X * norm_Z * att[2]
    sh_basis[:, 8] = np.sqrt(15) / 4 / np.sqrt(np.pi) * (norm_X ** 2 - norm_Y ** 2) * att[2]
    return sh_basis

def sh2lighting(sh, img_size=256):
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)

    x *= valid
    y *= valid
    z *= valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    shading = (shading * 255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    return shading

if __name__ == '__main__':
    shading = get_lighting(110, 65)
    print(shading.shape)
    plt.imshow(shading, cmap='gray')
    plt.show()
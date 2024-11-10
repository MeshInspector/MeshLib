from meshlib import mrmeshnumpy as mn
import numpy as np

u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

mesh = mn.meshFromUVPoints(x,y,z)

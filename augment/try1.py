import random
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import griddata
import numpy as np
from config import *


# img_sx = 295
# img_sy = 360
img_sx = 600
img_sy = 600
filename = "damier.jpg" # "actin.png"

def generate_deformation(n,sizex,sizey,k):
    # Les 4 points des extrémités ne sont pas modifié
    destination = np.array([[0, 0], [0, sizey], [sizex, 0], [sizex, sizey]])
    source = np.array([[0, 0], [0, sizey], [sizex, 0], [sizex, sizey]])
    for i in range(n):
        x = random.randint(i*round(sizex/n), (i+1)*round(sizex/n))
        y = random.randint(i*round(sizey/n), (i+1)*round(sizey/n))
        dx = random.randint(-k,k)
        dy = random.randint(-k, k)
        destination = np.append(destination,[[x+dx,y+dy]],axis=0)
        source = np.append(source ,[[x,y]],axis=0)
    return destination,source

grid_x, grid_y = np.mgrid[0:img_sx:img_sx*1j, 0:img_sy:img_sy*1j]
destination,source = generate_deformation(10,img_sx,img_sy,20)

grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(600,600)
map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(600,600)
map_x_32 = map_x.astype('float32')
map_y_32 = map_y.astype('float32')


orig = cv2.imread(test_folder_path+filename)
warped = cv2.remap(orig, map_x_32, map_y_32, cv2.INTER_CUBIC)
cv2.imwrite(test_folder_path+"/res.jpg", warped)

#PLOT ZONE
fig = plt.figure()
ax = fig.add_subplot(111)

# Grab some test data.


# Plot a basic wireframe.
# ax.plot_wireframe(grid_x, grid_y,[map_x,map_y], rstride=20, cstride=20)

ax.plot(map_x,map_y)
plt.show()
#FIN PLOT ZONE
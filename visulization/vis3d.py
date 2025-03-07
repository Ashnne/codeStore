import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def vis_3d(x_grid, y_grid, z):
    # z = np.sinc(np.sqrt(x_grid ** 2 + y_grid ** 2))
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(x_grid, y_grid, z, cmap=cm.viridis)
    plt.show()

def main():
    x = np.linspace(0, 1, 256)
    y = np.linspace(1, 0, 256)
    x_grid, y_grid = np.meshgrid(x, y)
    z = (x_grid - 0.5) ** 2 + (y_grid - 0.5) ** 2
    vis_3d(x_grid,y_grid,z)

if __name__ == "__main__":
    main()
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
from pdb import set_trace


class Wall():
    def __init__(self, width, thick, center=(0, 0)):
        self.w = width
        self.t = thick
        self.cx, self.cy = center
        self.x_min = self.cx - self.w
        self.x_max = self.cx + self.w
        self.y_min = self.cy - self.t
        self.y_max = self.cy + self.t
        self.orientation = 1

        if self.w > self.t:
            self.p1 = (self.x_min, self.cy)
            self.p2 = (self.x_max, self.cy)
            self.orientation = 1  # Vertical
        else:
            self.p1 = (self.cx, self.y_max)
            self.p2 = (self.cx, self.y_min)
            self.orientation = 0  # Horizontal


def potential_field_2d(x, y):
    dis = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    V = potential_field_1d(dis)
    return V


def potential_field_1d(x, d_0=1, alpha=0.2, d_1=1.8, d_2=0.01):
    d_0 = d_0
    alpha = alpha
    d_1 = d_1
    d_2 = d_2

    # if x <= d_2:
    #     V = alpha * d_0 / d_2
    # elif x > d_2 and x <= d_1:
    #     V = alpha * d_0 / x
    # else:
    #     V = alpha * d_0 / d_1

    if x < 0.4:
        V = 3
    else:
        V = 0

    return V


def potential_field_1d_force(x, d_0=1, alpha=10, d_1=10, d_2=0.01):
    d_0 = d_0
    alpha = alpha
    d_1 = d_1
    d_2 = d_2

    u = alpha*(1/x - d_0/math.pow(x, 2))
    u = np.max([u, -5])

    return u


def positivity(x):
    if x >= 0:
        return 1
    else:
        return -1


def potential_field_wall(x, y, Wall):
    t_num = (Wall.p1[0] - x)*(Wall.p2[0] - Wall.p1[0]) + \
        (Wall.p1[1] - y)*(Wall.p2[1] - Wall.p1[1])
    t_den = math.pow(Wall.p2[0] - Wall.p1[0], 2) + \
        math.pow(Wall.p2[1] - Wall.p1[1], 2)
    t = -t_num/t_den

    if 0 <= t and t <= 1:
        d_num = (Wall.p2[0] - Wall.p1[0])*(Wall.p1[1] - y) - \
            (Wall.p2[1] - Wall.p1[1])*(Wall.p1[0] - x)
        d_num = math.fabs(d_num)
        d_den = math.sqrt(t_den)
        d = d_num/d_den
        if Wall.orientation == 1:  # Veritcal Wall
            dx = 0
            dy = potential_field_1d(d) * positivity(y - Wall.p1[1])
        elif Wall.orientation == 0:  # Horizontal Wall
            dx = potential_field_1d(d) * positivity(x - Wall.p1[0])
            dy = 0
    else:
        d1 = math.pow((Wall.p2[0] - x), 2) + math.pow(Wall.p2[1] - y, 2)
        d2 = math.pow(Wall.p1[0] - x, 2) + math.pow(Wall.p1[1] - y, 2)
        if d1 < d2:
            d = math.sqrt(d1)
            dx = potential_field_1d(d) * math.fabs((Wall.p2[0] - x)/d) * positivity(x - Wall.p2[0])
            dy = potential_field_1d(d) * math.fabs((Wall.p2[1] - y)/d) * positivity(y - Wall.p2[1])
        else:
            d = math.sqrt(d2)
            dx = potential_field_1d(d) * math.fabs((Wall.p1[0] - x)/d) * positivity(x - Wall.p1[0])
            dy = potential_field_1d(d) * math.fabs((Wall.p1[1] - y)/d) * positivity(y - Wall.p1[1])

    return potential_field_1d(d), dx, dy, d


def potential_room(x, y):
    wall1 = Wall(0.5, 2, (0, 0))
    # print('wall1: ', wall1.orientation)
    wall2 = Wall(0.5, 2, (3, 0))
    # print('wall2: ', wall2.orientation)
    wall3 = Wall(1, 0.5, (2, -2))
    # print('wall3: ', wall3.orientation)
    wall4 = Wall(1, 0.5, (1, 2))
    # print('wall4: ', wall4.orientation)
    walls = [wall1, wall2, wall3, wall4]
    field = 0
    dx = 0
    dy = 0

    for w in walls:
        _field, _dx, _dy, _d = potential_field_wall(x, y, w)
        field += _field
        dx += _dx
        dy += _dy

    if field > 10:
        field = 10

    return field, dx, dy


def virtual_leader(x, y, l_x, l_y):
    # negative field values pushes the agent away and positive field values
    # attracts the agent
    dis_x = l_x - x
    dis_y = l_y - y
    dis = math.sqrt(math.pow(dis_x, 2) + math.pow(dis_y, 2))
    field = potential_field_1d_force(dis)

    dx = field * math.fabs(dis_x/dis) * positivity(l_x - x)
    dy = field * math.fabs(dis_y/dis) * positivity(l_y - y)

    _dx = 0
    _dy = 0
    # w = Wall(0.5, 2, (3, 0))
    # _, _dx, _dy, _ = potential_field_wall(x, y, w)
    _, _dx, _dy = potential_room(x, y)

    return field, dx+_dx, dy+_dy


def main():
    # Room
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            _, Z[j][i], _ = potential_room(X[i][j], Y[i][j])

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # Wall distance
    # wall = Wall(0.5, 2, (3, 0))
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    # X, Y = np.meshgrid(X, Y)
    # Z = np.zeros_like(X)
    # for i in range(X.shape[1]):
    #     for j in range(X.shape[0]):
    #         _, Z[j][i], dy, _ = potential_field_wall(Y[i][j], X[i][j], wall)
    #
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    #
    # # Customize the z axis.
    # ax.set_zlim(0, 10)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()


if __name__ == "__main__":
    main()

import math
import numpy as np
from utils import positivity
from utils import potential_field_1d_force
from utils import virtual_leader
from utils import virtual_obstacle


def edge_tension(edge_len_desired, x_robot, x_neighbor):
    delta = 2
    edge_len_current = x_robot - x_neighbor

    nmrtr = 2*(delta - math.fabs(edge_len_desired)) - \
        math.fabs(edge_len_current-edge_len_desired)
    dnmntr = delta - math.fabs(edge_len_desired) - \
        math.fabs(edge_len_current-edge_len_desired)
    dnmntr = math.pow(dnmntr, 2)
    alpha = x_robot - x_neighbor - edge_len_desired

    return (nmrtr/dnmntr) * alpha


def formation_control(type, msgs, id, r_pos):
    # Square Formation
    if type == "square":
        des_coord = 2*np.array([[0, -0.25], [0, 0], [0, 0.25],
                                [0.5, -0.25], [0.5, 0], [0.5, 0.25]])
    # Circle Formation
    if type == "circle":
        des_coord = 2*np.array([[-0.125, -math.sqrt(3)/8], [-0.25, 0],
                                [-0.125, math.sqrt(3)/8],
                                [0.125, -math.sqrt(3)/8],
                                [0.25, 0], [0.125, math.sqrt(3)/8]])
    # Diamond Formation
    if type == "diamond":
        des_coord = np.array([[0, 0], [2/3, 0], [4/3, 0],
                              [2, 0], [1, 1], [1, -1]])

    dx = 0
    dy = 0

    for msg in msgs:
        n_id = msg[0]
        n_pos = msg[1]

        des_x, des_y = des_coord[id] - des_coord[n_id]
        x_r, y_r, _ = r_pos
        x_n, y_n, _ = n_pos
        dx = dx - edge_tension(des_x, x_r, x_n)
        dy = dy - edge_tension(des_y, y_r, y_n)

        l_x, l_y, _ = r_pos - n_pos
        dis = math.sqrt(math.pow(l_x, 2) + math.pow(l_y, 2))
        u = potential_field_1d_force(dis, alpha=5, d_0=0.8)
        _dx = u * math.fabs(l_x/dis) * positivity(-l_x)
        _dy = u * math.fabs(l_y/dis) * positivity(-l_y)

        dx += _dx
        dy += _dy

    # Clip velocity
    lim = 50
    dx = np.clip(dx, -lim, lim)
    dy = np.clip(dy, -lim, lim)

    return dx, dy


def eject_from_ball(ball_x, ball_y, msgs, r_pos,
                    alpha=60, d_0=0.8, room_alpha=1):
    x, y, z = r_pos
    field, dx, dy = virtual_leader(x, y, ball_x, ball_y, alpha=alpha,
                                   d_0=d_0, room_alpha=room_alpha)

    for msg in msgs:
        n_pos = msg[1]
        l_x, l_y, _ = n_pos - r_pos
        dis = math.sqrt(math.pow(l_x, 2) + math.pow(l_y, 2))
        u = potential_field_1d_force(dis, alpha=3, d_0=0.7)

        _dx = u * math.fabs(l_x/dis) * positivity(l_x)
        _dy = u * math.fabs(l_y/dis) * positivity(l_y)

        dx += _dx
        dy += _dy

    # Clip velocity
    lim = 50
    dx = np.clip(dx, -lim, lim)
    dy = np.clip(dy, -lim, lim)

    return dx, dy


def follow_leader(leader_x, leader_y, r_pos, msgs,
                  leader_alpha=60, agent_alpha=10, agent_d_0=1):
    x, y, z = r_pos
    field, dx, dy = virtual_leader(x, y, leader_x, leader_y,
                                   alpha=leader_alpha)

    for msg in msgs:
        n_pos = msg[1]
        l_x, l_y, _ = n_pos - r_pos
        dis = math.sqrt(math.pow(l_x, 2) + math.pow(l_y, 2))
        u = potential_field_1d_force(dis, alpha=agent_alpha, d_0=agent_d_0)

        _dx = u * math.fabs(l_x/dis) * positivity(l_x)
        _dy = u * math.fabs(l_y/dis) * positivity(l_y)

        dx += _dx
        dy += _dy

    # Clip velocity
    lim = 50
    dx = np.clip(dx, -lim, lim)
    dy = np.clip(dy, -lim, lim)

    return dx, dy


def go_to_leader(leader_x, leader_y, r_pos, msgs,
                 obstacle_x, obstacle_y, obstacle=False, room_alpha=0):
    x, y, z = r_pos
    field, dx, dy = virtual_leader(x, y, leader_x, leader_y,
                                   alpha=60, room_alpha=room_alpha)

    if obstacle:
        field, _dx, _dy = virtual_obstacle(x, y, obstacle_x, obstacle_y)
        dx += _dx
        dy += _dy

    for msg in msgs:
        n_pos = msg[1]
        l_x, l_y, _ = n_pos - r_pos
        dis = math.sqrt(math.pow(l_x, 2) + math.pow(l_y, 2))
        u = potential_field_1d_force(dis, alpha=10)

        _dx = u * math.fabs(l_x/dis) * positivity(l_x)
        _dy = u * math.fabs(l_y/dis) * positivity(l_y)

        dx += _dx
        dy += _dy

    # Clip velocity
    lim = 50
    dx = np.clip(dx, -lim, lim)
    dy = np.clip(dy, -lim, lim)

    return dx, dy


def get_current_formation(msgs, id, r_pos):
    des_xs = np.zeros(6)
    des_ys = np.zeros(6)

    x, y, _ = r_pos
    des_xs[id] = x
    des_ys[id] = y

    for msg in msgs:
        n_id = msg[0]
        n_pos = msg[1]
        des_xs[n_id] = n_pos[0]
        des_ys[n_id] = n_pos[1]

    cx = np.mean(des_xs)
    cy = np.mean(des_ys)
    return [des_xs, des_ys, cx, cy]


def push_ball(msgs, id, r_pos, des_xs, des_ys, cx, cy, trgt_x, trgt_y, K=3):
    x, y, z = r_pos
    xs = np.zeros(6)
    ys = np.zeros(6)
    xs[id] = x
    ys[id] = y

    dx = 0
    dy = 0

    for msg in msgs:
        n_id = msg[0]
        n_pos = msg[1]
        xs[n_id] = n_pos[0]
        ys[n_id] = n_pos[1]

        des_x = des_xs[id] - des_xs[n_id]
        des_y = des_ys[id] - des_ys[n_id]
        l_x, l_y, _ = r_pos - n_pos

        x_r, y_r, _ = r_pos
        x_n, y_n, _ = n_pos
        dx = dx - edge_tension(des_x, x_r, x_n)
        dy = dy - edge_tension(des_y, y_r, y_n)

        dis = math.sqrt(math.pow(l_x, 2) + math.pow(l_y, 2))
        u = potential_field_1d_force(dis, alpha=1, d_0=1)

        _dx = u * math.fabs(l_x/dis) * positivity(-l_x)
        _dy = u * math.fabs(l_y/dis) * positivity(-l_y)

        dx += _dx
        dy += _dy

    cx = np.mean(xs)
    cy = np.mean(ys)

    K = 3
    fx = K * (trgt_x - cx)
    fy = K * (trgt_y - cy)

    dx += fx
    dy += fy

    return dx, dy, cx, cy

import math
from pdb import set_trace


def square_formation_control(d, l, x_r, x_n):
    delta = 2
    set_trace()
    nmrtr = 2*(delta - math.fabs(d)) - math.fabs(l-d)
    dnmntr = delta - math.fabs(d) - math.fabs(l-d)
    dnmntr = math.pow(dnmntr, 2)
    alpha = x_r - x_n - d

    return (nmrtr/dnmntr) * alpha


def main():
    d = -1
    ls = 9.416194883260509e-07
    x_r = 0.49748636930998874
    x_n = 0.4974854276905004

    dx = square_formation_control(d, ls, x_r, x_n)
    print(dx)


if __name__ == "__main__":
    main()

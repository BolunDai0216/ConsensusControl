import numpy as np
import math
from numpy.linalg import matrix_rank


def main():
    R = np.array([[-2, 0, 2, 0, 0, 0, 0, 0],
                  [0, 0, 0, 2, 0, -2, 0, 0],
                  [-2, 2, 0, 0, 2, -2, 0, 0],
                  [(math.sqrt(14)-2)/2, (math.sqrt(14)+2)/2, 0, 0,
                   0, 0, (2-math.sqrt(14))/2, -(math.sqrt(14)+2)/2],
                  [0, 0, 0, 0, (2+math.sqrt(14))/2, (math.sqrt(14)-2)/2, -(2+math.sqrt(14))/2, (2-math.sqrt(14))/2]])
    print("The rank for the rigidity matrix is {}".format(matrix_rank(R)))


if __name__ == "__main__":
    main()

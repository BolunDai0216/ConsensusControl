from utils import potential_field_1d_force
import matplotlib.pyplot as plt


def plot_1d_field():
    power = []
    xs = []

    for i in range(1000):
        tmp = (i+1)*0.01
        power.append(potential_field_1d_force(tmp, d_0=0.5))
        xs.append(tmp)

    return power, xs


def main():
    xs, power = plot_1d_field()
    plt.plot(power, xs)
    plt.show()


if __name__ == "__main__":
    main()

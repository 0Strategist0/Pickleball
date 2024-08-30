import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt


def main():
    DATA_PATH = "ball.csv"
    
    # Load the data from the falling
    falling_data = np.genfromtxt(DATA_PATH, float, delimiter=", ", skip_header=True)
    
    plt.scatter(falling_data[:, 0], falling_data[:, 1])
    plt.show()

if __name__ == "__main__":
    main()
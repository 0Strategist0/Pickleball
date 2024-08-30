import numpy as np
import numpy.typing as npt
import scipy.optimize as so
import scipy.integrate as si
import matplotlib.pyplot as plt

from pickleball import derivatives

def main():
    DATA_PATH = "ball.csv"
    
    # Load the data from the falling
    falling_data = np.genfromtxt(DATA_PATH, float, delimiter=", ", skip_header=True)
    
    # Fit the drag coefficient
    fit_drag_coef = so.curve_fit(time_to_ground, falling_data[:, 0], falling_data[:, 1], 0.05)[0]
    
    print(fit_drag_coef)
    
    plt.scatter(falling_data[:, 0], falling_data[:, 1])
    heights = np.linspace(falling_data[:, 0].min(), falling_data[:, 0].max(), 100)
    plt.plot(heights, time_to_ground(heights, fit_drag_coef))
    plt.show()

def time_to_ground(heights: float, drag_coef: float) -> float:
    
    T_MIN = 0.0
    T_MAX = 10.0
    X0 = 0.0
    V0 = 0.0
    ANGLE = 0.0
    WIND = 0.0
    GRAVITY = 9.81
    
    def hit_ground(_t: float, values: npt.ArrayLike, _drag_coef: float, _wind: float, gravity: float) -> float:
        """Return the y position of the pickleball for detection of ground collision

        Args:
            - _t (float): Time at which to return the y position. Not used for returning y position
            - values (npt.ArrayLike): The current values of [x position, x velocity, y position, y velocity]
            - _drag_coef (float): The coefficient C used in the calculation of the drag force Cv**2. Not used for returning y position
            - _wind (float): The wind velocity in the x direction. Not used for returning y position
            - _gravity (float): Gravitational acceleration. Not used for returning y position

        Returns:
            - float: The y position of the pickleball
        """
        
        return values[2]
    hit_ground.terminal = True
    
    return np.asarray([si.solve_ivp(derivatives, 
                                    (T_MIN, T_MAX), 
                                    (X0, V0 * np.cos(ANGLE), height, V0 * np.sin(ANGLE)), 
                                    dense_output=True, 
                                    events=hit_ground, args=(drag_coef, WIND, GRAVITY)).t_events[0][0] for height in heights])

if __name__ == "__main__":
    main()
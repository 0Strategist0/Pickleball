import numpy as np
import numpy.typing as npt
import scipy.integrate as si
import scipy.optimize as so
import matplotlib.pyplot as plt

def main() -> None:
    # Define constants
    NUM = {"x": 0, "vx": 1, "y": 2, "vy": 3}
    NAMES = ("x position (m)", "x velocity (m/s)", "y position (m)", "y velocity (m/s)")
    T_MIN = 0.0
    T_MAX = 10.0
    DRAG_COEF = 0.5 * (1.2) * (37.0 * 10.0 ** (-3.0)) ** 2.0 * np.pi * 0.6 / (24.0 * 10 ** (-3.0)) # 0.5 * rho * A * CD / m
    INITIAL_SPEED_GUESS = 10.0
    COURT_LENGTH = 13.4112
    GRAVITY = 9.81
    
    print(f"Drag Coefficient: {DRAG_COEF} 1/m")
    
    # Define input variables
    wind = 0.0
    x0 = 0.0
    y0 = 1.0
    angle = 0.5 * np.pi / 4.0
    opponent = 10.0 # Between 8.8392 and 13.4112
    
    # Solve the system 
    output = solve_system(x0, angle, y0, opponent, T_MIN, T_MAX, DRAG_COEF, wind, GRAVITY, COURT_LENGTH, INITIAL_SPEED_GUESS)
    
    # Plot results
    times_before_opponent = np.linspace(0.0, output.t_events[1][0], 100)
    times_after_opponent = np.linspace(output.t_events[1][0], output.t_events[0][0], 100)
    # Plot the ball trajectory
    plt.plot((output.sol(times_before_opponent)[NUM["x"]]), (output.sol(times_before_opponent)[NUM["y"]]), c="b")
    plt.plot((output.sol(times_after_opponent)[NUM["x"]]), (output.sol(times_after_opponent)[NUM["y"]]), c=(0.5,) * 3, ls=":")
    # Plot the court
    plt.axhline(0.0, c="k")
    plt.axvline(0.0, c="k")
    plt.axvline((COURT_LENGTH), c="k")
    plt.plot((COURT_LENGTH / 2.0, COURT_LENGTH / 2.0), (0.0, f2m(3.0)), c="k")
    # Show collision with opponent
    plt.scatter((output.y_events[1][0][0]), (output.y_events[1][0][2]), s=25, c="k", zorder=10)
    plt.text((output.y_events[1][0][0]), (output.y_events[1][0][2]), f"  {output.t_events[1][0]:.2} seconds")
    # Formatting
    plt.xlabel(NAMES[NUM["x"]])
    plt.ylabel(NAMES[NUM["y"]])
    plt.title(f"Ball Trajectory (Wind = {wind:.0f} m/s)")
    plt.show()
    
    # Plot speed over time
    plt.plot(times_before_opponent, 
             np.sqrt(output.sol(times_before_opponent)[NUM["vx"]] ** 2.0 
                     + output.sol(times_before_opponent)[NUM["vy"]] ** 2.0), c="b")
    plt.plot(times_after_opponent, 
             np.sqrt(output.sol(times_after_opponent)[NUM["vx"]] ** 2.0 
                     + output.sol(times_after_opponent)[NUM["vy"]] ** 2.0), c=(0.5,) * 3, ls=":")
    plt.xlabel("Time (s)")
    plt.ylabel("Ball Speed (m/s)")
    plt.title(f"Ball Speed Over Time (Wind = {wind:.0f} m/s)")
    plt.show()
    
    # Plot speed relative to wind over time
    plt.plot(times_before_opponent, 
             np.sqrt((output.sol(times_before_opponent)[NUM["vx"]] - wind) ** 2.0 
                     + output.sol(times_before_opponent)[NUM["vy"]] ** 2.0), c="b")
    plt.plot(times_after_opponent, 
             np.sqrt((output.sol(times_after_opponent)[NUM["vx"]] - wind) ** 2.0 
                     + output.sol(times_after_opponent)[NUM["vy"]] ** 2.0), c=(0.5,) * 3, ls=":")
    plt.xlabel("Time (s)")
    plt.ylabel("Ball Speed Relative to Air (m/s)")
    plt.title(f"Ball Speed Relative to Air Over Time (Wind = {wind:.0f} m/s)")
    plt.show()

def solve_system(x0: float, 
                 angle: float, 
                 y0: float, 
                 opponent: float, 
                 tmin: float, 
                 tmax: float, 
                 drag_coef: float, 
                 wind: float, 
                 gravity: float, 
                 court_length: float, 
                 inital_speed_guess: float) -> si._ivp.ivp.OdeResult:
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
    
    def reached_opponent(_t: float, values: npt.ArrayLike, _drag_coef: float, _wind: float, gravity: float) -> float:
        """Return the x offset of the pickleball from the opponent for detecting opponent collision

        Args:
            - _t (float): Time at which to return the y position. Not used for returning x offset
            - values (npt.ArrayLike): The current values of [x position, x velocity, y position, y velocity]
            - _drag_coef (float): The coefficient C used in the calculation of the drag force Cv**2. Not used for returning x offset
            - _wind (float): The wind velocity in the x direction. Not used for returning x offset
            - _gravity (float): Gravitational acceleration. Not used for returning x offset

        Returns:
            - float: The x offset of the pickleball from the opponent
        """
        
        return values[0] - opponent
    
    # Find velocity where ground touch is at end of court
    optimal_speed = so.fsolve(lambda v: si.solve_ivp(derivatives, 
                                                     (tmin, tmax), 
                                                     (x0, v[0] * np.cos(angle), y0, v[0] * np.sin(angle)), 
                                                     events=hit_ground, 
                                                     args=(drag_coef, wind, gravity)).y_events[0][0][0] - court_length, 
                              inital_speed_guess)[0]
    
    print(f"Optimal Speed: {optimal_speed} m/s")
    
    # Solve the differential equation with that velocity
    return si.solve_ivp(derivatives, 
                        (tmin, tmax), 
                        (x0, optimal_speed * np.cos(angle), y0, optimal_speed * np.sin(angle)), 
                        dense_output=True, 
                        events=[hit_ground, reached_opponent], args=(drag_coef, wind, gravity))

def derivatives(_t: float, current_values: npt.ArrayLike, drag_coef: float, wind: float, gravity: float) -> npt.ArrayLike:
    """Calculate the derivatives of x position, x velocity, y position, and y velocity. 

    Args:
        - _t (float): Time at which to calculate the derivatives. Not used as system is time-invariant
        - current_values (npt.ArrayLike): The current values of [x position, x velocity, y position, y velocity] 
                                        for which to calculate the derivatives
        - drag_coef (float): The coefficient C used in the calculation of the drag force Cv**2
        - wind (float): The wind velocity in the x direction
        - gravity (float): Gravitational acceleration

    Returns:
        - npt.ArrayLike: x velocity, x accleration, y velocity, y acceleration
    """
    
    x_derivative = current_values[1]
    vx_derivative = -drag_coef * (current_values[1] - wind) * np.sqrt((current_values[1] - wind) ** 2.0 + current_values[3] ** 2.0)
    y_derivative = current_values[3]
    vy_derivative = -gravity - drag_coef * current_values[3] * np.sqrt((current_values[1] - wind) ** 2.0 + current_values[3] ** 2.0)
    
    return (x_derivative, vx_derivative, y_derivative, vy_derivative)

def f2m(feet: float) -> float:
    """Convert a distance in metres to a distance in feet

    Args:
        feet (float): Distance in feet

    Returns:
        float: Distance in metres
    """
    
    return feet / 3.28084

def m2f(metres: float) -> float:
    """Convert a distance in metres to a distance in feet

    Args:
        metres (float): Distance in metres

    Returns:
        float: Distance in feet
    """
    
    return metres * 3.28084

if __name__ == "__main__":
    main()
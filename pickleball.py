import numpy as np
import numpy.typing as npt
import scipy.integrate as si
import scipy.optimize as so
import matplotlib.pyplot as plt
# from mayavi import mlab
from tqdm import tqdm

def main() -> None:
    # Define constants
    T_MIN = 0.0
    T_MAX = 10.0
    DRAG_COEF = 0.5 * (1.2) * (37.0 * 10.0 ** (-3.0)) ** 2.0 * np.pi * 0.6 / (24.0 * 10 ** (-3.0)) # 0.5 * rho * A * CD / m
    INITIAL_SPEED_GUESS = 10.0
    COURT_LENGTH = f2m(44.0)
    GRAVITY = 9.81
    N_RANDOM_SAMPLES = 1000
    
    print(f"Drag Coefficient: {DRAG_COEF} 1/m")
    
    # Plot the time difference histogram
    time_difference_plot(N_RANDOM_SAMPLES, T_MIN, T_MAX, DRAG_COEF, GRAVITY, COURT_LENGTH, INITIAL_SPEED_GUESS)
    
    # ax.set_xlabel("Initial Position (m)")
    # ax.set_ylabel("Launch Angle (rad)")
    # ax.set_zlabel("Time to Opponent (s)")
    # ax.legend()
    # plt.show()
    
    # # Plot results
    # times_before_opponent = np.linspace(0.0, output.t_events[1][0], 100)
    # times_after_opponent = np.linspace(output.t_events[1][0], output.t_events[0][0], 100)
    # # Plot the ball trajectory
    # plt.plot((output.sol(times_before_opponent)[NUM["x"]]), (output.sol(times_before_opponent)[NUM["y"]]), c="b")
    # plt.plot((output.sol(times_after_opponent)[NUM["x"]]), (output.sol(times_after_opponent)[NUM["y"]]), c=(0.5,) * 3, ls=":")
    # # Plot the court
    # plt.axhline(0.0, c="k")
    # plt.axvline(0.0, c="k")
    # plt.axvline((COURT_LENGTH), c="k")
    # plt.plot((COURT_LENGTH / 2.0, COURT_LENGTH / 2.0), (0.0, f2m(3.0)), c="k")
    # # Show collision with opponent
    # plt.scatter((output.y_events[1][0][0]), (output.y_events[1][0][2]), s=25, c="k", zorder=10)
    # plt.text((output.y_events[1][0][0]), (output.y_events[1][0][2]), f"  {output.t_events[1][0]:.2} seconds")
    # # Formatting
    # plt.xlabel(NAMES[NUM["x"]])
    # plt.ylabel(NAMES[NUM["y"]])
    # plt.title(f"Ball Trajectory (Wind = {wind:.0f} m/s)")
    # plt.show()
    
    # # Plot speed over time
    # plt.plot(times_before_opponent, 
    #          np.sqrt(output.sol(times_before_opponent)[NUM["vx"]] ** 2.0 
    #                  + output.sol(times_before_opponent)[NUM["vy"]] ** 2.0), c="b")
    # plt.plot(times_after_opponent, 
    #          np.sqrt(output.sol(times_after_opponent)[NUM["vx"]] ** 2.0 
    #                  + output.sol(times_after_opponent)[NUM["vy"]] ** 2.0), c=(0.5,) * 3, ls=":")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ball Speed (m/s)")
    # plt.title(f"Ball Speed Over Time (Wind = {wind:.0f} m/s)")
    # plt.show()
    
    # # Plot speed relative to wind over time
    # plt.plot(times_before_opponent, 
    #          np.sqrt((output.sol(times_before_opponent)[NUM["vx"]] - wind) ** 2.0 
    #                  + output.sol(times_before_opponent)[NUM["vy"]] ** 2.0), c="b")
    # plt.plot(times_after_opponent, 
    #          np.sqrt((output.sol(times_after_opponent)[NUM["vx"]] - wind) ** 2.0 
    #                  + output.sol(times_after_opponent)[NUM["vy"]] ** 2.0), c=(0.5,) * 3, ls=":")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ball Speed Relative to Air (m/s)")
    # plt.title(f"Ball Speed Relative to Air Over Time (Wind = {wind:.0f} m/s)")
    # plt.show()
    
        # Solve the system 
    # print(solve_system(0.0, 1.0, y0, opponent, T_MIN, T_MAX, DRAG_COEF, wind, GRAVITY, COURT_LENGTH, INITIAL_SPEED_GUESS).t_events[1][0])
    
    # GRID_SHAPE = (15, 15)
    # x0s, angles = np.meshgrid(np.linspace(0, 4.572, GRID_SHAPE[0]), np.linspace(0.0, np.pi / 4.0, GRID_SHAPE[1]))
    # # x0s, angles = np.meshgrid(np.linspace(0, 4.572, GRID_SHAPE[0]), np.linspace(8.8392, 13.4112, GRID_SHAPE[1]))
    # x0s = x0s.T
    # angles = angles.T
    # # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # figure = mlab.figure("Time to Opponent")
    
    # for wind, color, label in zip((-5.0, 0.0, 5.0), ((1.0, 0.0, 0.0), (0.5,) * 3, (0.0, 0.0, 1.0)), ("Against the Wind", "No Wind", "With the Wind")):
    #     times_to_opponent = np.asarray([solve_system(x0, 
    #                                                 angle, 
    #                                                 y0, 
    #                                                 opponent, 
    #                                                 T_MIN, 
    #                                                 T_MAX, 
    #                                                 DRAG_COEF, 
    #                                                 wind, 
    #                                                 GRAVITY, 
    #                                                 COURT_LENGTH, 
    #                                                 INITIAL_SPEED_GUESS).t_events[1][0] 
    #                                     for x0, angle in zip(x0s.ravel(), angles.ravel())]).reshape(GRID_SHAPE)
        
    #     # Plot 3D
    #     surface = mlab.surf(x0s, angles, times_to_opponent, color=color)
    #     # surface = ax.plot_surface(x0s, angles, times_to_opponent, color=color, label=label)
    
    # mlab.axes(xlabel="Initial Position", ylabel="Launch Angle", zlabel="Time to Opponent")
    # mlab.show()

def time_difference_plot(n_samples: int, 
                         tmin: float, 
                         tmax: float, 
                         drag_coef: float, 
                         gravity: float, 
                         court_length: float, 
                         initial_speed_guess: float) -> None:
    """Plot the difference in time-to-opponent for a variety of wind speeds

    Args:
        n_samples (int): The number of random samples to calculate time for
        tmin (float): The start time of the ODE solver
        tmax (float): The end time of the ODE solver
        drag_coef (float): The drag coefficient
        gravity (float): Gravitational acceleration
        court_length (float): The length of the court
        initial_speed_guess (float): The initial speed to guess when solving
    """
    
    # Initialize with random parameters
    wind_rand = np.random.uniform(0.0, 15.0, n_samples)
    x0_rand = np.random.uniform(0.0, f2m(15.0), n_samples)
    y0_rand = np.random.uniform(0.0, f2m(7.0), n_samples)
    angle_rand = np.random.uniform(np.deg2rad(0.0), np.deg2rad(30.0), n_samples)
    opponent_rand = np.random.uniform(f2m(29.0), f2m(44.0), n_samples)
    
    # Loop through and find times for all random params
    times = np.zeros(n_samples)
    for index, (wind, x0, y0, angle, opponent) in tqdm(enumerate(zip(wind_rand, x0_rand, y0_rand, angle_rand, opponent_rand))):
        time_with_wind = solve_system(x0, 
                                      angle, 
                                      y0, 
                                      opponent, 
                                      tmin, 
                                      tmax, 
                                      drag_coef, 
                                      wind, 
                                      gravity, 
                                      court_length, 
                                      initial_speed_guess).t_events[1]
        
        time_against_wind = solve_system(x0, 
                                         angle, 
                                         y0, 
                                         opponent, 
                                         tmin, 
                                         tmax, 
                                         drag_coef, 
                                         -wind, 
                                         gravity, 
                                         court_length, 
                                         initial_speed_guess).t_events[1]
        
        times[index] = time_with_wind[0] - time_against_wind[0] if len(time_with_wind) > 0 and len(time_against_wind) > 0 else np.nan
    
    print(f"{np.count_nonzero(np.isnan(times))} NaN events")
    
    plt.hist2d(wind_rand[np.isfinite(times)], times[np.isfinite(times)], bins=20)
    plt.xlabel("Wind Speed")
    plt.ylabel("Time Difference Playing With Wind Versus Against Wind")
    plt.title("Difference in Time-To-Opponent Playing With and\nAgainst Wind for Various Wind Speeds")
    plt.colorbar()
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

def mps2mph(metres_per_second: float) -> float:
    """Convert a speed in metres per second to a speed in miles per hour

    Args:
        metres_per_second (float): Speed in m/s

    Returns:
        float: Speed in mph
    """
    
    return metres_per_second * 2.23694

def mph2mps(metres_per_second: float) -> float:
    """Convert a speed in miles per hour to a speed in metres per second

    Args:
        metres_per_second (float): Speed in mph

    Returns:
        float: Speed in m/s
    """
    
    return metres_per_second / 2.23694

if __name__ == "__main__":
    main()
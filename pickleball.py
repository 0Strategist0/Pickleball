import numpy as np
import numpy.typing as npt
import scipy.integrate as si
import scipy.optimize as so
import matplotlib.pyplot as plt
# from mayavi import mlab
from tqdm import tqdm
from itertools import cycle

# Initialize linestyle cycler
linestyle_cycler = cycle(['-','--',':','-.'])

def main() -> None:
    # Define constants
    T_MIN = 0.0
    T_MAX = 10.0
    DRAG_COEF = 0.5 * (1.2) * (37.0 * 10.0 ** (-3.0)) ** 2.0 * np.pi * 0.6 / (24.0 * 10 ** (-3.0)) # 0.5 * rho * A * CD / m
    INITIAL_SPEED_GUESS = 10.0
    COURT_LENGTH = f2m(44.0)
    GRAVITY = 9.81
    N_RANDOM_SAMPLES = 10000
    KDE_STDEV = 2.0
    MAX_WIND = 15.0
    
    print(f"Drag Coefficient: {DRAG_COEF} 1/m")
    
    # # Plot the time difference histogram
    # time_difference_plot(N_RANDOM_SAMPLES, MAX_WIND, T_MIN, T_MAX, DRAG_COEF, GRAVITY, COURT_LENGTH, INITIAL_SPEED_GUESS, KDE_STDEV)
    
    # # Plot the initial speed histogram
    # velocity_plot(N_RANDOM_SAMPLES, MAX_WIND, T_MIN, T_MAX, DRAG_COEF, GRAVITY, COURT_LENGTH, INITIAL_SPEED_GUESS, KDE_STDEV)
    
    # Plot some trajectories
    trajectory_plot(x0_list=[f2m(11.0)], 
                    angle_list=[np.deg2rad(20.0)], 
                    y0_list=[f2m(3.0)], 
                    opponent=f2m(35.0), 
                    wind_list=(mph2mps(-10.0), mph2mps(0.0), mph2mps(10.0), mph2mps(15.0)), 
                    tmin=T_MIN, 
                    tmax=T_MAX, 
                    drag_coef=DRAG_COEF, 
                    gravity=GRAVITY, 
                    court_length=COURT_LENGTH, 
                    initial_speed_guess=INITIAL_SPEED_GUESS)
    
    # # Get the time-to-opponent for various initial parameters
    # for x0 in [f2m(0.0), f2m(11.0), f2m(15.0)]:
    #     for y0 in [f2m(3.0)]:
    #         for angle in [np.deg2rad(20.0)]:
    #             for opponent in [f2m(29.0), f2m(33.0), f2m(44.0)]:
    #                 for wind in [mph2mps(10.0), mph2mps(15.0), mph2mps(20.0)]:
    #                     pos = solve_system(x0, 
    #                                        angle, 
    #                                        y0, 
    #                                        opponent, 
    #                                        T_MIN, 
    #                                        T_MAX, 
    #                                        DRAG_COEF, 
    #                                        wind, 
    #                                        GRAVITY, 
    #                                        COURT_LENGTH, 
    #                                        INITIAL_SPEED_GUESS)
    #                     neg = solve_system(x0, 
    #                                        angle, 
    #                                        y0, 
    #                                        opponent, 
    #                                        T_MIN, 
    #                                        T_MAX, 
    #                                        DRAG_COEF, 
    #                                        -wind, 
    #                                        GRAVITY, 
    #                                        COURT_LENGTH, 
    #                                        INITIAL_SPEED_GUESS)
                        
    #                     print(f"x0={m2f(x0)}',y0={m2f(y0)}',theta={np.rad2deg(angle)}°,z0={m2f(opponent)}',wind={mps2mph(wind)}mph: "
    #                           + (str(pos.t_events[1][0] - neg.t_events[1][0]) if len(pos.t_events[1]) > 0 and len(neg.t_events[1]) > 0 
    #                              else str(pos.t_events[0][0] - neg.t_events[0][0])))
    
    


def kde(x: npt.ArrayLike, points: npt.ArrayLike, stdev: float) -> npt.ArrayLike:
    points = np.asarray(points)
    
    return (np.sum(points[1, np.newaxis] * np.exp(-0.5 * ((points[0, np.newaxis] - x[:, np.newaxis]) / stdev) ** 2.0), axis=1)
            / np.sum(np.exp(-0.5 * ((points[0, np.newaxis] - x[:, np.newaxis]) / stdev) ** 2.0), axis=1))

def trajectory_plot(x0_list: npt.ArrayLike, 
                    angle_list: npt.ArrayLike, 
                    y0_list: npt.ArrayLike, 
                    opponent: float, 
                    tmin: float, 
                    tmax: float, 
                    drag_coef: float, 
                    wind_list: npt.ArrayLike, 
                    gravity: float, 
                    court_length: float, 
                    initial_speed_guess: float) -> None:
    
    def plot_one_position(x0, angle, y0, wind):
        output = solve_system(x0, angle, y0, opponent, tmin, tmax, drag_coef, wind, gravity, court_length, initial_speed_guess)
        
        # Label trajectory
        label = ""
        if len(x0_list) > 1:
            label += f"Initial Position = {m2f(x0):.0f} feet, "
        if len(angle_list) > 1:
            label += f"Launch Angle = {np.rad2deg(angle):.0f}°, "
        if len(y0_list) > 1:
            label += f"Initial Height = {m2f(y0):.0f} feet, "
        if len(wind_list) > 1:
            label += f"Wind Speed = {mps2mph(wind):.0f} mph, "
        label = label.strip(", ")
        
        # Plot results
        times = np.linspace(0.0, output.t_events[0][0], 100)
        
        # Plot the ball trajectory
        plt.plot(m2f(output.sol(times)[0]), m2f(output.sol(times)[2]), label=label, ls=next(linestyle_cycler))
    
    # Set up the trajectory figure
    fig = plt.figure(figsize=(12.0, 4.0))
    
    for x0 in x0_list:
        for angle in angle_list:
            for y0 in y0_list:
                for wind in wind_list:
                    plot_one_position(x0, angle, y0, wind)
    
    # Plot the court
    plt.axhline(0.0, c="k")
    plt.axvline(0.0, c="k")
    plt.axvline(m2f(court_length), c="k")
    plt.plot((m2f(court_length / 2.0), m2f(court_length / 2.0)), (0.0, 3.0), c="k")
    
    plt.xlabel("Horizontal Position (feet)")
    plt.ylabel("Vertical Position (feet)")
    
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlim(0.0, 44.0)
    
    plt.savefig(f"Trajectory_x0{x0_list}_y0{y0_list}_angle{angle_list}_wind{wind_list}.pdf")
    
    
    def plot_one_speed(x0, angle, y0, wind):
        output = solve_system(x0, angle, y0, opponent, tmin, tmax, drag_coef, wind, gravity, court_length, initial_speed_guess)
        
        # Label trajectory
        label = ""
        if len(x0_list) > 1:
            label += f"Initial Position = {m2f(x0):.0f} feet, "
        if len(angle_list) > 1:
            label += f"Launch Angle = {np.rad2deg(angle):.0f}°, "
        if len(y0_list) > 1:
            label += f"Initial Height = {m2f(y0):.0f} feet, "
        if len(wind_list) > 1:
            label += f"Wind Speed = {mps2mph(wind):.0f} mph, "
        label = label.strip(", ")
        
        # Plot results
        times = np.linspace(0.0, output.t_events[0][0], 100)
        
        # Plot the ball trajectory
        plt.plot(m2f(output.sol(times)[0]), mps2mph(np.sqrt(output.sol(times)[1] ** 2.0 + output.sol(times)[3] ** 2.0)), label=label, ls=next(linestyle_cycler))
    
    # Set up the velocity figure
    fig = plt.figure()
    
    for x0 in x0_list:
        for angle in angle_list:
            for y0 in y0_list:
                for wind in wind_list:
                    plot_one_speed(x0, angle, y0, wind)
    
    plt.xlabel("Horizontal Position (feet)")
    plt.ylabel("Speed (mph)")
    
    plt.legend()
    
    plt.savefig(f"Speed_x0{x0_list}_y0{y0_list}_angle{angle_list}_wind{wind_list}.pdf")

def time_difference_plot(n_samples: int, 
                         max_wind: float,
                         tmin: float, 
                         tmax: float, 
                         drag_coef: float, 
                         gravity: float, 
                         court_length: float, 
                         initial_speed_guess: float, 
                         kde_stdev: float) -> None:
    """Plot the difference in time-to-opponent for a variety of wind speeds

    Args:
        n_samples (int): The number of random samples to calculate time for
        max_wind (float): The maximum wind speed to plot
        tmin (float): The start time of the ODE solver
        tmax (float): The end time of the ODE solver
        drag_coef (float): The drag coefficient
        gravity (float): Gravitational acceleration
        court_length (float): The length of the court
        initial_speed_guess (float): The initial speed to guess when solving
        kde_stdev (float): The standard deviation to use for kernel density estimation
    """
    
    # Initialize with random parameters
    wind_rand = np.random.uniform(0.0, max_wind, n_samples)
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
    
    plt.hist2d(mps2mph(wind_rand[np.isfinite(times)]), times[np.isfinite(times)], bins=20)
    wind_range = np.linspace(0.0, max_wind)
    plt.plot(mps2mph(wind_range), 
             kde(wind_range, (wind_rand[np.isfinite(times)], times[np.isfinite(times)]), kde_stdev), 
             c="cyan", 
             label="Average Time Difference")
    plt.xlabel("Wind Speed (mph)")
    plt.ylabel("Time Difference Playing With Wind Versus Against Wind (s)")
    plt.title("Difference in Time-To-Opponent Playing With and\nAgainst Wind for Various Wind Speeds")
    cbar = plt.colorbar()
    cbar.set_label("Number of Hits")
    plt.legend()
    plt.show()

def velocity_plot(n_samples: int, 
                  max_wind: float,
                  tmin: float, 
                  tmax: float, 
                  drag_coef: float, 
                  gravity: float, 
                  court_length: float, 
                  initial_speed_guess: float, 
                  kde_stdev: float) -> None:
    """Plot the difference in time-to-opponent for a variety of wind speeds

    Args:
        n_samples (int): The number of random samples to calculate time for
        max_wind (float): The maximum wind speed to plot
        tmin (float): The start time of the ODE solver
        tmax (float): The end time of the ODE solver
        drag_coef (float): The drag coefficient
        gravity (float): Gravitational acceleration
        court_length (float): The length of the court
        initial_speed_guess (float): The initial speed to guess when solving
        kde_stdev (float): The standard deviation to use for kernel density estimation
    """
    
    # Initialize with random parameters
    wind_rand = np.random.uniform(-max_wind, max_wind, n_samples)
    x0_rand = np.random.uniform(0.0, f2m(15.0), n_samples)
    y0_rand = np.random.uniform(0.0, f2m(7.0), n_samples)
    angle_rand = np.random.uniform(np.deg2rad(0.0), np.deg2rad(30.0), n_samples)
    
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
    
    # Loop through and find times for all random params
    velocities = np.zeros(n_samples)
    for index, (wind, x0, y0, angle) in tqdm(enumerate(zip(wind_rand, x0_rand, y0_rand, angle_rand))):
        velocities[index] = so.fsolve(lambda v: si.solve_ivp(derivatives, 
                                                             (tmin, tmax), 
                                                             (x0, v[0] * np.cos(angle), y0, v[0] * np.sin(angle)), 
                                                             events=hit_ground, 
                                                             args=(drag_coef, wind, gravity)).y_events[0][0][0] - court_length, 
                              initial_speed_guess)[0]
    
    plt.hist2d(mps2mph(wind_rand[np.isfinite(velocities)]), np.log10(mps2mph(velocities[np.isfinite(velocities)])), bins=20)
    wind_range = np.linspace(-max_wind, max_wind)
    plt.plot(mps2mph(wind_range), 
             kde(wind_range, (wind_rand[np.isfinite(velocities)], np.log10(mps2mph(velocities[np.isfinite(velocities)]))), kde_stdev), 
             c="cyan", 
             label="Average Hit Speed")
    plt.xlabel("Wind Speed (mph)")
    plt.ylabel("Log of Hit Speed (log10(mph))")
    plt.title("Hit Speed Required at Different Wind Speeds")
    cbar = plt.colorbar()
    cbar.set_label("Number of Hits")
    plt.legend()
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
                 initial_speed_guess: float, 
                 return_speed: bool = False) -> si._ivp.ivp.OdeResult | tuple[si._ivp.ivp.OdeResult, float]:
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
                              initial_speed_guess)[0]
    
    # Solve the differential equation with that velocity
    if not return_speed:
        return si.solve_ivp(derivatives, 
                            (tmin, tmax), 
                            (x0, optimal_speed * np.cos(angle), y0, optimal_speed * np.sin(angle)), 
                            dense_output=True, 
                            events=[hit_ground, reached_opponent], args=(drag_coef, wind, gravity))
    else:
        return (si.solve_ivp(derivatives, 
                             (tmin, tmax), 
                             (x0, optimal_speed * np.cos(angle), y0, optimal_speed * np.sin(angle)), 
                             dense_output=True, 
                             events=[hit_ground, reached_opponent], args=(drag_coef, wind, gravity)), 
                optimal_speed)

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
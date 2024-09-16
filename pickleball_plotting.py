"""Pickleball Plotting
Kye Emond
Sept 2024

A file to plot some trajectories of an optimally-hit pickleball given quadratic drag. 
"""

import numpy as np
import matplotlib.pyplot as plt

from pickleball_solving import solve_system

def main() -> None:
    # Define constants
    T_MIN = 0.0
    T_MAX = 10.0
    DRAG_COEF = 0.5 * (1.2) * (37.0 * 10.0 ** (-3.0)) ** 2.0 * np.pi * 0.6 / (24.0 * 10 ** (-3.0)) # 0.5 * rho * A * CD / m
    INITIAL_SPEED_GUESS = 10.0
    COURT_LENGTH = f2m(44.0)
    GRAVITY = 9.81
    
    # Plot some trajectories
    for wind in (mph2mps(15.0), mph2mps(0.0), mph2mps(-15.0)):
        trajectory_plot(x0=f2m(11.0), 
                        angle=np.deg2rad(20.0), 
                        y0=f2m(3.0), 
                        opponent=f2m(30.0), 
                        wind=wind, 
                        tmin=T_MIN, 
                        tmax=T_MAX, 
                        drag_coef=DRAG_COEF, 
                        gravity=GRAVITY, 
                        court_length=COURT_LENGTH, 
                        initial_speed_guess=INITIAL_SPEED_GUESS)

def trajectory_plot(x0: float, 
                    angle: float, 
                    y0: float, 
                    opponent: float, 
                    tmin: float, 
                    tmax: float, 
                    drag_coef: float, 
                    wind: float, 
                    gravity: float, 
                    court_length: float, 
                    initial_speed_guess: float) -> None:
    """Plot the trajectories and velocities of an optimally-hit pickleball. 

    Args:
        x0 (float): The initial x position of the ball
        angle (float): The launch angle of the ball
        y0 (float): The initial y position of the ball
        opponent (float): The position of the opponent
        tmin (float): The starting integration time
        tmax (float): The maximum allowed integration time
        drag_coef (float): The drag coefficient (note: not the same as C_D)
        wind (float): The wind speed
        gravity (float): The gravitational acceleration
        court_length (float): The length of the court
        initial_speed_guess (float): The launch speed of the ball
    """
    
    output = solve_system(x0, angle, y0, opponent, tmin, tmax, drag_coef, wind, gravity, court_length, initial_speed_guess)
    
    # Plot results
    times_before_opponent = np.linspace(0.0, output.t_events[1][0], 100)
    times_after_opponent = np.linspace(output.t_events[1][0], output.t_events[0][0], 100)
    
    # Set up the trajectory figure
    plt.figure()
    # Plot the ball trajectory
    plt.plot(m2f(output.sol(times_before_opponent)[0]), m2f(output.sol(times_before_opponent)[2]), c="b")
    plt.plot(m2f(output.sol(times_after_opponent)[0]), m2f(output.sol(times_after_opponent)[2]), c=(0.5,) * 3, ls=":")
    # Plot the court
    plt.axhline(0.0, c="k")
    plt.axvline(0.0, c="k")
    plt.axvline(m2f(court_length), c="k")
    plt.plot((m2f(court_length / 2.0), m2f(court_length / 2.0)), (0.0, 3.0), c="k")
    # Show collision with opponent
    plt.scatter(m2f(output.y_events[1][0][0]), m2f(output.y_events[1][0][2]), s=25, c="k", zorder=10)
    plt.text(m2f(output.y_events[1][0][0]), m2f(output.y_events[1][0][2]), f"  {output.t_events[1][0]:.2} seconds")
    # Formatting
    plt.xlabel("Horizontal Position (feet)")
    plt.ylabel("Vertical Position (feet)")
    plt.savefig(f"Trajectory_wind={mps2mph(wind):.0f}mph,x0={m2f(x0):.0f}ft,z0={m2f(opponent):.0f}ft,"
                + f"angle={np.rad2deg(angle):.0f}deg.png")
    
    # Set up the velocity figure
    plt.figure()
    # Plot the velocity against time
    plt.plot(times_before_opponent, mps2mph(np.sqrt(output.sol(times_before_opponent)[1] ** 2.0 
                                                    + output.sol(times_before_opponent)[3] ** 2.0)))
    plt.plot(times_after_opponent, mps2mph(np.sqrt(output.sol(times_after_opponent)[1] ** 2.0 
                                                   + output.sol(times_after_opponent)[3] ** 2.0)), c=(0.5,) * 3, ls=":")
    # Show collision with opponent
    plt.scatter(output.t_events[1][0], mps2mph(np.sqrt(output.y_events[1][0][1] ** 2.0 + output.y_events[1][0][3] ** 2.0)), 
                s=25, c="k", zorder=10)
    plt.text(output.t_events[1][0], mps2mph(np.sqrt(output.y_events[1][0][1] ** 2.0 + output.y_events[1][0][3] ** 2.0)), 
             f"  {output.t_events[1][0]:.2} seconds")
    # Formatting
    plt.xlabel("Time (seconds)")
    plt.ylabel("Speed (mph)")
    plt.savefig(f"Velocity_wind={mps2mph(wind):.0f}mph,x0={m2f(x0):.0f}ft,z0={m2f(opponent):.0f}ft,"
                + f"angle={np.rad2deg(angle):.0f}deg.png")

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
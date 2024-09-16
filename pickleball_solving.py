"""Pickleball Solving
Kye Emond
Sept 2024

A module that solves for the projectile motion of an optimally-hit pickleball given quadratic drag. 

**Functions**:
- solve_system: Returns the OdeResult object from Scipy's solve_ivp function
- derivatives: Calculates the derivative vector of (x, x', y, y')
"""

import numpy as np
import numpy.typing as npt
import scipy.integrate as si
import scipy.optimize as so

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
    """Return the solution to the pickleball trajectory equations

    Args:
        - x0 (float): The initial x position of the ball
        - angle (float): The launch angle of the ball
        - y0 (float): The initial y position of the ball
        - opponent (float): The position of the opponent
        - tmin (float): The starting integration time
        - tmax (float): The maximum allowed integration time
        - drag_coef (float): The drag coefficient to use for calculation (note, different from C_D)
        - wind (float): The wind speed
        - gravity (float): The gravitational acceleration
        - court_length (float): The length of the pickleball court
        - initial_speed_guess (float): The launch speed of the ball
        - return_speed (bool, optional): Whether to return the optimal launch speed as well as the solution. Defaults to False.

    Returns:
        - si._ivp.ivp.OdeResult | tuple[si._ivp.ivp.OdeResult, float]: The solution to the differential equation, optionally
            including the optimal launch speed
    """
    
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
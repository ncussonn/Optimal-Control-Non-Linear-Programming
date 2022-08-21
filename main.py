from re import U
from time import time
from tkinter import N
import numpy as np
from utils import *
from casadi import *

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5        #1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k (lissajous is a type of curve)
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller - does not function well
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

'''PART 1: Receding-Horizon Certainty Equivalent Control (CEC)'''
def CEC(cur_state, ref_state, cur_iter):

    ''' TUNING PARAMETERS '''
    T = 50                 # number of control intervals (max 100)
    Q = 7 *MX.eye(2)        # control weighting - tune
    R = 4 *MX.eye(2)        # performance weighting - tune 
    gamma = 0.9999          # Try to get as close to 1 as possible
    q = 10                  # scalar that defines how much error associated with deviating from proper angle
    terminal_cost = 0       # 

    # Using Opti

    opti = Opti()

    E = opti.variable(3,T+1)    # error matrix
    U = opti.variable(2,T)      # control matrix

    V_star = terminal_cost

    for t in range(T):

        p_tilde = E[:2,t]
        theta_tilde = E[2,t]
        u = U[:,t]

        V_star += gamma**t * (p_tilde.T @ Q @ p_tilde + q*(1-cos(theta_tilde))**2 + u.T @ R @ u)

        cur_reference_traj  = lissajous(cur_iter + t)
        next_reference_traj = lissajous(cur_iter + t + 1)

        # Getting Next Position
        x_new = E[0,t] + cur_reference_traj[0]
        y_new = E[1,t] + cur_reference_traj[1]

        # APPLYING CONSTRAINTS

        # g constraint (Motion Model)
        opti.subject_to(E[:,t+1] == motionModel(cur_reference_traj,next_reference_traj, E[:,t], U[:,t]))    # Must motion model

        # Position Constraints
        opti.subject_to(opti.bounded(-3,x_new,3))
        opti.subject_to(opti.bounded(-3,y_new,3))

        # Free Space Constraint
        buffer = 0.05
        opti.subject_to((x_new + 2)**2 + (y_new + 2)**2 >= 0.5**2 + buffer)       # for circle of center (-2,-2) and radius 0.5
        opti.subject_to((x_new - 1)**2 + (y_new - 2)**2 >= 0.5**2 + buffer)       # for circle of center ( 1, 2) and radius 0.5

    # Constraints
    opti.subject_to(opti.bounded(0, U[0,:], 1))     # lin velocity
    opti.subject_to(opti.bounded(-1, U[1,:], 1))    # ang velocity

    current_error = cur_state - ref_state

    opti.subject_to(E[:,0] == current_error)        # initial error must be same as current

    # Initial Solver Guesses
    opti.set_initial(E[:,0], current_error)         # doesn't do anything because constraint, but needed for solver
    opti.set_initial(U[:,0], [0, 0])                # any control will do within bounds

    # Goal of Objective
    opti.minimize(V_star)                           # want to minimize the value function 

    # Solve Non-Linear Program (NLP)
    suppress = {'ipopt.print_level':0, 'print_time':0}  # suppresses unneccesary output
    opti.solver("ipopt", suppress)                      # Solve problem
    opti_variables = opti.solve()

    return opti_variables.value(U[:,0])     # return first control action to follow


'''PART 2: Generalized Policy Iteration (GPI)'''
def GPI(cur_state, ref_state):

    std_dev = np.array([0.04, 0.04, 0.04]).T

    # Unfinished Code Using PI
    '''
    def policyEvaluation(X,V,T,policy):

        V = np.array([[V[1]],[V[2]]])
        I = np.identity(2)

        # Precise
        V_N = np.linalg.inv(I-P_NN) @ (P_NT @ q)

        V = np.zeros((8,1))
        for i in range(len(X)):
            x = X[i]
            V[i] = terminalCost(x)

        V[1] = V_N[0]
        V[2] = V_N[1]

        #print('V: {}'.format(V))

        return V    # returns array

    def policyImprovement(x,U,X,T,V):

        new_policy_dict = {}
        for u in U:
            sigma = []
            print(u)
            for x_prime in X:
                sigma.append(motionModel(x,u,x_prime,T)*V[X.index(x_prime)])

            summation = sum(sigma)
            new_policy_dict[u] = summation

        #print('New Policy Dictionary: {}'.format(new_policy_dict))
        new_policy = min(new_policy_dict, key=new_policy_dict.get)

        return new_policy

    # Using Precise Policy Evaluation

    # Initialize V0
    V0 = np.zeros((8,1))
    pi = {10000:('RED',10000),20000:('RED',10000)}
    V = policyEvaluation(X,V0,T,pi)
    #print('V_prior Before While: {}'.format(V_prior))
    k = 0
    epsilon = 1

    while epsilon >= 0.1: 
        
        pi = {}  
        V_prior = V     

        for x in N:
            pi[x] = policyImprovement(x,U,X,T,V)

        V = policyEvaluation(X,V,T,pi)
        print('V_new: {}'.format(V))
        print('V_prior: {}'.format(V_prior))

        k+=1
        print('Iteration: {}'.format(k))

        epsilon = np.linalg.norm(V - V_prior)
        print('Epsilon: {}'.format(epsilon))
    
    print('Improved Policy: {}'.format(pi))
    '''

    v = 0
    w = 0

    return [v,w]

# This function implements the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()

if __name__ == '__main__':

    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []

    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0

    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        '''REPLACE CONTROLLER HERE'''
        # Generate control input
        
        # DEMO CONTROL
        #control = simple_controller(cur_state, cur_ref, next_ref)
        
        # PART A CONTROL
        control = CEC(cur_state,cur_ref,cur_iter)

        # PART B CONTROL:
        #control = GPI(cur_state, cur_ref)

        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)  # noise = False: removes noise
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state - cur_ref)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)
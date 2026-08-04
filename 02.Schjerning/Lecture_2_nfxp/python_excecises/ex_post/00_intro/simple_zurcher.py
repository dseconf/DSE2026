# import
import numpy as np
from scipy import interpolate

def cost(x, par):
    return par.c * x 

def util(d, x, par):
    return -cost(x,par) - par.RC*(d==1)

def setup():   
    """
    Function initializes and sets up the parameters and grids for the simple zurcher model.
    
    Returns:
      the parameter object `par` which contains various parameters and grids used in the code.
    """

    class par: pass

    # 1. structual parameters
    par.RC = 11.7257 # Replacement cost
    par.c = 2.45569 * 0.001 # Cost parameter
    par.beta = 0.99 # Discount factor
    
    # 2. parameters for VFI
    par.max_iter = 10000 # maximum number of iterations
    par.tol = 1.0e-8 #convergence tol. level 
    
    # 3. mileage grid
    par.n = 175 # Number of grid points
    par.max = 450 # Max of mileage
    par.grid = np.arange(0,par.n) # milage grid
    
    # 4. create transition arrays (exercise 2)
    par.transition = np.array([0,1,2]) # Transition grid
    par.p = np.array([1/3, 1/3, 1/3]) # Transition probability grid
    
    # 5. draw extreme value type 1 / Gumbel taste shocks (exercise 3)
    np.random.seed(1987)
    par.num_eps = 10000
    par.sigma_eps = 1.0
    par.eps_keep_gumb = np.random.gumbel(loc=-par.sigma_eps * np.euler_gamma,scale=par.sigma_eps,size=par.num_eps)
    par.eps_replace_gumb = np.random.gumbel(loc=-par.sigma_eps * np.euler_gamma, scale=par.sigma_eps,size=par.num_eps)
    
    # 6. draw standard normal taste shocks (exercise 5)
    par.eps_keep_norm = np.random.normal(0,1,par.num_eps)
    par.eps_replace_norm = np.random.normal(0,1,par.num_eps)
    
    # 7. call function to make choice-dependent markov (transition) matrices
    par.P1 = create_transition_matrix(0, par)
    par.P2 = create_transition_matrix(1, par)
        
    return par

def solve_SA(par, vectorized = False, **kwargs):
    """
    The function `solve_SA` performs successive approximations (VFI)
    The function creates a class 'sol', containing:
    - `V`: a numpy array representing the value function
    - `pk`: a numpy array representing the policy function (conditional choice probabilities)
    - `it`: an integer representing the number of iterations performed
    - `delta`: a float representing the maximal difference between the current and previous value functions
    """
    
    # 1. create class
    class sol: pass

    # 2. allocate
    sol.V = np.zeros([par.n]) # arbitrary starting values
    sol.pk = np.zeros([par.n]) # arbitrary starting values

    # 3. initialize
    sol.it = 0
    sol.delta = 2000
        
    # 4. iterate
    while (sol.it <= par.max_iter and sol.delta > par.tol):
        
        # a. compute Bellman
        if vectorized==False:
            V_now, pk = bellman(sol.V, par, **kwargs)

        else:
            V_now, pk = bellman_vector(sol.V, par, **kwargs)

        # b. update class
        sol.delta = np.amax(np.abs(V_now - sol.V))
        sol.it += 1
        sol.V = V_now
        sol.pk = pk

    # 5. print
    print(f'Finished after {sol.it} iterations')
    print(f'Convergence achieved: {sol.delta < par.tol}')
    
    return sol

def bellman(V_next, par, taste_shocks = 'None', stochastic_transition = False):
    """
    Function evaluates the integrated value bellmann-operator in a dynamic programming problem.
    For a given guess of the (integrated) value function for the next period, it calculates a new guess
    of the (integrated) value funtion and also the choice probabilities. The function is made general 
    for with and without taste-shocks, stochastic state-transition, etc.
    """
    
    # 1. initialize
    V_now = np.zeros([par.n])
    pk = np.zeros([par.n])
    
    # 2. loop over state grid
    for x in par.grid:
            
        # a. compute expected future value for each choice, given the state: EV(x,d)
        if stochastic_transition == False:
            EV_keep = EV_deterministic(0, x, V_next, par)
            EV_replace = EV_deterministic(1, x, V_next, par)

        else:
            EV_keep = EV_stochastic(0, x, V_next, par)
            EV_replace = EV_stochastic(1, x, V_next, par)

        # b. Calculate value of each choice, "value-choice functions"
        value_keep = util(0, x, par) + par.beta*EV_keep
        value_replace = util(1, x, par) + par.beta*EV_replace
        maxV = np.amax([value_keep, value_replace])
        
        # 3. Find the maximum value across choices -> the (integrated) value function
        if taste_shocks == 'None':
            V_now[x] = maxV
            pk[x] = (value_replace < value_keep) # either 0 or 1
        
        elif taste_shocks == 'Analytical: Extreme Value':
            exp_keep = np.exp( (value_keep-maxV)/par.sigma_eps )
            exp_replace = np.exp( (value_replace-maxV)/par.sigma_eps )
            V_now[x] = (maxV + par.sigma_eps * np.log(exp_keep+exp_replace)) # see trick in slides
            pk[x] = exp_keep/(exp_keep+exp_replace)
            
        elif taste_shocks == 'Monte Carlo: Extreme Value':
            values = np.column_stack([value_keep + par.eps_keep_gumb, value_replace + par.eps_replace_gumb]) # shape (numeps, 2)
            choices = np.argmax(values, axis = 1)

            V_now[x] = np.mean(np.max(values, axis=1)) # 1) take max of each row 2) take mean of all maxes'
            pk[x] = 1 - choices.mean()
        
        elif taste_shocks == 'Monte Carlo: Normal':
            values = np.column_stack([value_keep + par.eps_keep_norm, value_replace + par.eps_replace_norm]) # shape (numeps, 2)
            choices = np.argmax(values, axis = 1)
            V_now[x] = np.mean(np.max(values, axis=1)) # 1) take max of each row 2) take mean of all maxes'
            pk[x] = 1 - choices.mean()

    return V_now, pk

def EV_deterministic(d, x, V_next, par):
    """
    A function that computes next-period expected value function conditional on choice d: EV(x,d),
    where the expectation is wrt. the state variable, i.e. no stochasticity here
    """

    x_next = 1 + x*(d==0)
    x_next = np.fmin(x_next, par.n-1) # Ensure that x_next is within grid. par.n-1 is largest grid point
    EV = V_next[x_next]

    return EV

def EV_stochastic(d, x, V_next, par):
    """
    A function that computes next-period expected value function conditional on choice d: EV(x,d),
    where the expectation is wrt. the state variable
    """

    EV = 0
    x_next = x*(d==0)
    for prob_m, m in zip(par.p, par.transition):
        x_next_m = x_next+m
        x_next_m = np.fmin(x_next_m, par.n-1)
        EV += prob_m*V_next[x_next_m]
    
    return EV

def bellman_vector(V_next, par):
    """
    Function calculates the value and choice probabilities for each choice in a
    dynamic programming problem using the Bellman equation, vectorized
    """
    
    # Hint: The function should follow the structure of "bellman" but you have to code
    # up some of the auxilary functions that we use, e.g. EV_stochastic
    # structure of function "bellman":
    # 1) compute EV(x,d) using Markov (transition) matrices
    # 2) compute value of each choice, "value-choice functions"
    # 3) compute value function by features of Gumbel distribution
    # 1. compute expected future value across states for each choice
    EV_keep = par.P1 @ V_next # (n,n) x (n,) -> shape (n,)
    EV_replace = par.P2 @ V_next # (n,n) x (n,) -> shape (n,)

    # 2. compute value of each choice, "value-choice functions"
    value_keep = util(0, par.grid, par) + par.beta*EV_keep # (n,) + (n,) -> shape (n,)
    value_replace = util(1, par.grid, par) + par.beta*EV_replace # (n,) + (n,) -> shape (n,)

    # 3. compute value function by features of Gumbel distribution
    maxV = np.amax(np.column_stack([value_keep, value_replace]), axis=1)
    exp_keep = np.exp( (value_keep-maxV)/par.sigma_eps )
    exp_replace = np.exp( (value_replace-maxV)/par.sigma_eps )
    V_now = (maxV + par.sigma_eps * np.log(exp_keep+exp_replace))
    pk = exp_keep/(exp_keep+exp_replace)
            
    return V_now, pk

def create_transition_matrix(d, par):
    """
    The function `create_transition_matrix` creates a transition matrix based on the given parameters,
    where the transition probabilities depend on the choice.
    
    Args:
      d: The parameter "d" is used to determine the type of transition matrix to create. If "d" is 0, it
    means we want to create a transition matrix for the "keep" choice. If "d" is 1, it means we want
    to create a transition matrix for the 'replace' choice.
      par: The parameter `par` is an object that contains various parameters.

    
    Returns:
      the transition matrix, which is a numpy array of shape (n, n).
    """

    # 1. initialize Markov (transition) matrix
    P = np.zeros((par.n,par.n)) 
    
    # 2. keep
    if d == 0:
        for i in range(par.n): # loop over rows
            if i <= par.n-len(par.p): # check if p vector fits entirely
                P[i,i:i+len(par.p)]=par.p # slice row i and columns i till i + 3 (length of vector of shocks m)
            else: # truncate probabilities: cant go "outside" state grid
                P[i,i:] = par.p[:par.n-len(par.p)-i]
                P[i,-1] = 1.0-P[i][:-1].sum()

    # 3. replace
    if d == 1:
        for i in range(par.n): # loop over rows
            P[i][:len(par.p)]=par.p # fill in probabilities
    
    return P

import numpy as np
import matplotlib.pyplot as plt

#%% Define time-dependent rate functions.
def alpha_2(t, params):
    return params['alpha_2']

def tau_m(t, params):
    return params['tau_m']

#%% Define the model ODE system.
def meningitis_model(y, t, params):
    S, I_m, V_mp, R = y
    N = S + I_m + V_mp + R  # Total population

    a2 = alpha_2(t, params)
    tm = tau_m(t, params)

    kappa   = params['kappa']
    Lambda  = params['Lambda']
    rho     = params['rho']
    phi     = params['phi']
    mu      = params['mu']
    gamma_m = params['gamma_m']

    infection_force = a2 * I_m / N

    dS   = (1 - kappa)*Lambda + rho*R + phi*V_mp - (infection_force + mu)*S
    dI_m = infection_force * S - (tm + gamma_m + mu)*I_m
    dV_mp = Lambda * kappa - (mu + phi)*V_mp
    dR   = tm * I_m - (rho + mu)*R

    return np.array([dS, dI_m, dV_mp, dR])

#%% RK4 integrator for one time step.
def rk4_step(f, y, t, dt, params):
    k1 = f(y, t, params)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, params)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, params)
    k4 = f(y + dt*k3, t + dt, params)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

#%% Run the simulation.
def run_simulation(params, y0, t_span, dt):
    t0, tf = t_span
    t_vals = np.arange(t0, tf+dt, dt)
    sol = np.zeros((len(t_vals), len(y0)))
    sol[0] = y0
    for i in range(1, len(t_vals)):
        sol[i] = rk4_step(meningitis_model, sol[i-1], t_vals[i-1], dt, params)
    return t_vals, sol

#%% Main simulation script.
if __name__ == '__main__':
    
    # Base parameters
    base_params = {
        'Lambda':   0.5,    
        'kappa':    0.105,  
        'rho':      0.1,    
        'phi':      0.263,  
        'mu':       0.5,    
        'gamma_m':  0.05,   
        'tau_m':    0.02    
    }
    
    # Values of alpha_2 to test
    alpha_values = [0.007, 0.02, 0.1, 0.3, 0.7]

    # Compute Disease-Free Equilibrium (DFE)
    V_mp_eq = base_params['Lambda'] * base_params['kappa'] / (base_params['mu'] + base_params['phi'])
    S_eq = ((1 - base_params['kappa'])*base_params['Lambda'] + base_params['phi'] * V_mp_eq) / base_params['mu']
    R_eq = 0.0
    perturb = 1e-4  # Small perturbation

    # Time settings
    t_span = (0, 200)  
    dt = 0.1           

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for a2 in alpha_values:
        params = base_params.copy()
        params['alpha_2'] = a2

        y0 = np.array([S_eq, perturb, V_mp_eq, R_eq])

        t_vals, sol = run_simulation(params, y0, t_span, dt)

        # Determine whether this is DFE or EE
        if a2 <= 0.02:
            label = f"DFE (α₂ = {a2})"
            axes[0].plot(t_vals, sol[:, 1], label=label)  # Plot I_m over time
        else:
            label = f"EE (α₂ = {a2})"
            axes[1].plot(t_vals, sol[:, 1], label=label)  # Plot I_m over time

    # Configure DFE plot
    axes[0].set_title("DFE Stability")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Infected Population (I_m)")
    axes[0].legend()
    axes[0].grid(True)

    # Configure EE plot
    axes[1].set_title("Endemic Equilibrium (EE) Stability")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Infected Population (I_m)")
    axes[1].legend()
    axes[1].grid(True)

    # Save both plots in one image
    plt.tight_layout()
    plt.savefig('2DFE_and_EE_stability.jpeg', format='jpeg')
    plt.show()

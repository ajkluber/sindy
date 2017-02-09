import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy

import util

def brownian_dynamics(dVdx, x0, n_steps,  mass=1.0, damping=1.0, dt=0.005, kT=1.0, boundary_map=None):
    coeff_A = dt / (mass * damping)       # this is related to the diffusion coefficient
    coeff_B = np.sqrt(2.0 * dt * kT / (mass * damping))
    x = [x0]
    for _i in range(n_steps):
        newx = x[-1] - coeff_A*dVdx(x[-1]) + coeff_B*np.random.normal()
        if not (boundary_map is None):
            newx = boundary_map(newx)
        x.extend(newx)
    xtraj = np.array(x, dtype=np.float64)
    return xtraj

def periodic_gradient():
    x = sympy.symbols("x")
    symV = 0.5*sympy.cos(x)
    symdVdx = sympy.diff(symV)
    dVdx = sympy.lambdify(x, symdVdx)
    return dVdx

def boundary(x):
    if x > 2.*np.pi:
        x -= 2.*np.pi
    elif x < 0:
        x += 2.*np.pi
    return x


if __name__ == "__main__":
    n_steps = 2**23
    x0 = 0.2*np.pi
    dim = 1
    L = 2*np.pi
    #centers = np.linspace(-1, 5, 100)
    bin_edges = np.linspace(0, L, 101)
    mid_bin  = 0.5*(bin_edges[1:] + bin_edges[:-1])

    # Create library of basis functions in symbolic algebra form.
    x = sympy.symbols('x')

    #database = [1, x, x**2, x**3]
    #database = [1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, sympy.sin(x), sympy.cos(x), sympy.exp(-x), sympy.exp(-x**2)]
    database = [ sympy.sin(k*x) for k in range(1,10) ] + [ sympy.cos(k*x) for k in range(10) ]
    basis_funcs = sympy.lambdify(x, database)

    # generate trajectories
    dVdx = periodic_gradient()
    xtraj = brownian_dynamics(dVdx, x0, n_steps, boundary_map=boundary)

    #n, bin_edges = np.histogram(md_trajs[0][:, 0], bins=bin_edges)

    # Subsampling trajectory smoothes energy features.
    step = 1
    dt = 0.005
    delta = step*dt
    idx = 0
    #traj = extract_traj(md_trajs, idx, step)

#    # Bin along coordinate to calculate the average force. ToDo: Discard poorly sampled bins.
#    bin_means, bin_edges, binnumber = stats.binned_statistic(xtraj[:-1], Y, statistic='mean', bins=100)
#    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
#
#    # solve weighted least squares problem in the bins
#    # see https://onlinecourses.science.psu.edu/stat501/node/352
#    n, bin_edges = np.histogram(md_trajs[0][:-1,0], bins=bin_edges)
#    n = n.astype(float)
#    W = n/np.sum(n)
#
#    Xred = compute_database_1d(bin_centers, basis_funcs)
#    Xred = Xred.transpose()
#    wXred = np.multiply(Xred, W[:,np.newaxis])
#    wbin_means = np.multiply(bin_means, W)
#
#    # Solve for basis function coefficients
#    iter_coef, iter_force, iter_err, iter_lamb = iterative_solver_regularized(wXred, W*bin_means, 10, bin_centers, basis_funcs)

    # Prepare basis function matric X and output Y (for a given value of nstep)
    X = np.array(map(basis_funcs, xtraj))

    # Calculate time derivative. In brownian dynamics this is an effective force.
    # deal with peridic boundary
    x_dot_temp = xtraj[1:] - xtraj[:-1]
    jumps = (L - np.abs(x_dot_temp)) < np.abs(x_dot_temp)
    x_dot_temp[jumps] = -np.sign(x_dot_temp[jumps])*(L - np.abs(x_dot_temp[jumps]))
    Y = x_dot_temp

    iter_coef, iter_force, iter_err, iter_lamb = util.iterative_solver_regularized(X, Y, 10, bin_centers, basis_funcs)

    plt.figure()
    #plt.plot(bin_centers, (iter_force[::2].T))
    for i in range(iter_force.shape[0]):
        plt.plot(bin_centers, iter_force[i], label=str(i))
    plt.legend()
    plt.plot(bin_centers, bin_means, 'k', lw=2)
    plt.xlabel("x")
    plt.ylabel("Force")

    #plt.figure()
    #plt.plot(iter_err)
    #plt.xlabel("Iteration")
    #plt.ylabel("Error")

    plt.figure()
    plt.plot(iter_lamb, label="$\\lambda$")
    plt.plot([ np.min(np.abs(temp[temp != 0.0])) for temp in iter_coef ], label="$min \\xi$")
    plt.xlabel("Iteration")
    plt.ylabel("Lambda")
    plt.legend()

    plt.figure()
    plt.pcolormesh(iter_coef)
    plt.xlabel("Coef")
    plt.ylabel("Iteration")
    plt.colorbar()

    plt.show()

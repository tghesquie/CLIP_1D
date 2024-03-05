import numpy as np

from simulation_parameters import SimulationParameters
from bulk_damage import BulkDamage
from model import Model
from solve import Solver
from dumper import Dumper


def run(parameters):

    N_elements = parameters.N_elements
    epsilon_0 = parameters.epsilon_0
    L = parameters.L
    wc = parameters.wc
    N_increments = parameters.N_increments

    # Initialize the model
    u = np.zeros(2 * N_elements)
    bc = {0                     : 0,
          (2 * N_elements) - 1  : 0}
    
    d_prev = np.zeros(N_elements -1)
    d = np.zeros(N_elements - 1)
    lambda_ = np.zeros(N_elements - 1)
    print(epsilon_0*L, wc*1.5, N_increments-1)

    # Define the increments
    incs = np.concatenate((np.array([0]), np.linspace(epsilon_0*L, wc*1.5, N_increments-1)))

    # Initialize the model and solver
    model = Model(parameters)
    solver = Solver(model, parameters)
    dumper = Dumper()

    for i, u_t in enumerate(incs):

        # Update the boundary conditions at the bar end
        bc[(2*N_elements)-1] = u_t

        # Introduce an artificial damage at the middle of the bar
        if i == 2:
            print("Introducing an artificial damage")
            d_prev = np.zeros(N_elements - 1)
            d_prev[int((N_elements-1)/2)] = 0.01
        else:
            d_prev = d.copy()

        print(" ------------------ ", i)

        results = solver.solve_functional(d, d_prev, bc)
        d = results.x
        D = solver.bulk_damage.get_Bulk_damage(d_prev)
        
        # Store the results
        u_, F_, lambda_, d_ = solver.equilibrium_solver.solve_equilibrium_ul(d, D, bc)
        functional = solver.functional.assemble_clip_functional(d, D, u_, lambda_, bc)
        dumper.store("imposed_displacement", u_t)
        dumper.store("force", F_[-1])
        dumper.store("displacement", u_)
        dumper.store("cohesive_damage", d)
        dumper.store("bulk_damage", D)
        dumper.store("lagrange", lambda_)
        dumper.store("cohesive_stress", parameters.k * solver.functional.get_jump(u_) * model.gd_cohesive(d))


    return dumper

if __name__ == '__main__':

    E = 3e10  # Young's Modulus (Pa)
    Gc = 120  # Fracture Energy (N/m)
    sigc = 3e6  # Stress Limit (N/m^2)
    L = 0.2  # Length of the bar (m)
    Dm = 0.7  # Bulk Damage parameter
    N_increments = 30  # Number of increments
    max_iter = 100  # Maximum number of iterations
    a = np.pi / 6  # G function - Co-efficients
    gamma = 0.5

    parameters = SimulationParameters(E, Gc, sigc, L, Dm, N_increments, max_iter, a, gamma)

    results = run(parameters)
    print(results.data["cohesive_stress"])

    # Plot imposed displacement vs cohesive stress
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(results.data["imposed_displacement"], results.data["force"])
    plt.xlabel("Imposed displacement (m)")
    plt.ylabel("Cohesive stress (N/m^2)")
    plt.show()

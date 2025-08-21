import pygmo as pg
import pandas as pd
import numpy as np
import sys
from numba import jit
import time

# On Windows, limit mp_island process pool size to avoid the HANDLE limit of WaitForMultipleObjects (64 handles)
try:
    # Initialise a global pool of 60 processes (<=61 to allow auxiliary handles)
    pg.mp_island.init_pool(processes=60)
except Exception:
    pass

# JIT-compiled fitness evaluator (no Python object references)
@jit(nopython=True)
def fitness_nb(dv, x_data, y_data):
    p = dv[0]
    q = dv[1]
    m = dv[2]
    rss = 0.0
    for i in range(x_data.shape[0]):
        x = x_data[i]
        num = 1.0 - (p + q)
        denom = 1.0 + (p + q) + 1e-10
        p_safe = p + 1e-10
        term1 = (1.0 - (num/denom)**(x + 1) / 2.0) / (1.0 + (q / p_safe) * ((num/denom)**(x + 1) / 2.0))
        term2 = (1.0 - (num/denom)**(x - 1) / 2.0) / (1.0 + (q / p_safe) * ((num/denom)**(x - 1) / 2.0))
        pred = m * (term1 - term2)
        diff = y_data[i] - pred
        rss += diff * diff
    return rss

class PyGMOBassProblem:
    def __init__(self, x, y):
        # assume x and y are numpy.float64 arrays
        self.x_data = x
        self.y_data = y


    def fitness(self, dv):
        # Call the jitted function and return a 1-element list for PyGMO
        rss = fitness_nb(dv, self.x_data, self.y_data)
        return [rss]

    def get_bounds(self):
        # p and q between 0 and 1
        return ([0.0, 0.0, 1.0], [1.0, 1.0, 100000.0])

def main(output_filename, bass_input_name):
    # Read input data
    bass_input = pd.read_csv(bass_input_name)

    # PSO settings
    # PSO_GENERATIONS = 5000
    # POPULATION_SIZE = 2000
    PSO_GENERATIONS = 5000
    POPULATION_SIZE = 2000

    results = []

    for zone in bass_input['ZoneID'].unique():
        print(time.ctime())
        df_zone = bass_input[bass_input['ZoneID'] == zone]
        x = df_zone.loc[df_zone['months_passed_01_2021'] > 0, 'months_passed_01_2021'].values.astype(np.float64)
        y = df_zone.loc[df_zone['months_passed_01_2021'] > 0, '2021-2024'].values.astype(np.float64)
        if x.size == 0 or y.size == 0:
            print(f"No data for zone {zone}, skipping.")
            continue

        # Set up optimization problem
        problem = pg.problem(PyGMOBassProblem(x, y))
        algo = pg.algorithm(pg.pso(gen=PSO_GENERATIONS))
        archi = pg.archipelago(n=64, algo=algo, prob=problem, pop_size=POPULATION_SIZE)
        archi.evolve()
        archi.wait()

        champions_x = archi.get_champions_x()
        champions_f = archi.get_champions_f()
        best_idx = np.argmin(champions_f)
        p_opt, q_opt, m_opt = champions_x[best_idx]

        # Record results
        results.append({'ZoneID': zone, 'p': p_opt, 'q': q_opt, 'm': m_opt})
        print(f"Zone {zone}: p={p_opt:.6f}, q={q_opt:.6f}, m={m_opt:.2f}")

    # Save results (p and q only)
    results_df = pd.DataFrame(results)
    out_file = f'best_parameter_{output_filename}.csv'
    results_df.to_csv(out_file, index=False)
    print(f"Best parameters (p, q) saved to '{out_file}'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Bass_Model_Parallelized_numba.py <market_size_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
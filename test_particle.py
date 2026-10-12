import time
from app.solvers.particle import solve_particle_box

t0 = time.perf_counter()
r = solve_particle_box(L_nm=2.0, n_states=5, n_grid=256)
dt = (time.perf_counter() - t0) * 1000

print(f"compute_time_ms = {dt:.3f}")
for i, E in enumerate(r["energies_eV"], 1):
    E_analytic = (i**2 * 6.62607015e-34**2) / (8 * 9.1093837015e-31 * (2e-9)**2) / 1.602176634e-19
    print(f"  E_{i} = {E:.6f} eV   (analytic: {E_analytic:.6f})")

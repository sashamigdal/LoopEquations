"""Physical units per Max Planck run from the data files themselves (SI: S2 in m^2/s^2, eps in m^2/s^3):
  u'^2 = S2(inf)/2,  nu from Re_lambda = u'^2 sqrt(15/(nu eps))  [isotropic Taylor scale]  and, independently,
  from the dissipation range S2 -> sqrt(eps nu)/15 (r/eta)^2;  eta = (nu^3/eps)^(1/4).
Then the fitted shift gives the diffusion length l_D = eta e^s and the fitted k_min gives the width W = pi l_D / k_min."""
import json, numpy as np
F = json.load(open('../mpi/fit_box.json'))
rows = []
for f in F:
    d = np.genfromtxt('../mpi/E_Kohler/'+f['file'], delimiter=',', skip_header=1)
    rn, S2 = d[:, 1], d[:, 2]
    eps = float(f['file'].split('Eps_')[1].rstrip('.csv').rstrip('.'))
    Re = f['Re']; u2 = S2[-5:].mean()/2
    nu_Re = 15*u2**2/(eps*Re**2)
    c = (S2[:3]/rn[:3]**2).mean()                     # -> sqrt(eps nu)/15 in the dissipation range
    nu_diss = (15*c)**2/eps
    nu = nu_Re; eta = (nu**3/eps)**0.25
    lD = eta*np.exp(f['s_box']); W = np.pi*lD/f['kmin'] if f['kmin'] > 1e-3 else np.inf
    lD_inf = eta*np.exp(f['s_inf'])
    rows.append(dict(Re=Re, eps=eps, u_rms=np.sqrt(u2), nu_Re=nu_Re, nu_diss=nu_diss, eta=eta, rmax_m=rn[-1]*eta,
                     lD_m=lD, lD_inf_m=lD_inf, kmin=f['kmin'], W_m=W))
    print('Re=%5.0f eps=%.3f u\'=%.3f m/s  nu(Re)=%.2e nu(diss)=%.2e m2/s  eta=%.1f um  r_max=%.2f m  l_D=%.3f m  k_min=%.2f  W=%.2f m' % (
        Re, eps, np.sqrt(u2), nu_Re, nu_diss, eta*1e6, rn[-1]*eta, lD, f['kmin'], W))
json.dump(rows, open('../mpi/physical_units.json', 'w'), indent=1)

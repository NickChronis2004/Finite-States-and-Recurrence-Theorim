import numpy as np
import matplotlib.pyplot as plt

# --- Helper functions ---
def random_hermitian(D, rng):
    X = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    H = (X + X.conj().T) / 2.0
    evals = np.linalg.eigvalsh(H)
    width = evals.max() - evals.min()
    if width > 0:
        H = H / width
    return H

def fidelity(psi0, psit):
    return np.abs(np.vdot(psi0, psit))**2

def partial_trace_rho(rho, dims, keep):
    rho_t = rho.reshape(*dims, *dims)
    n = len(dims)
    trace_out = sorted([i for i in range(n) if i not in keep], reverse=True)
    current_n = n
    for ax in trace_out:
        rho_t = np.trace(rho_t, axis1=ax, axis2=ax + current_n)
        current_n -= 1
    d_keep = int(np.prod([dims[i] for i in keep]))
    return rho_t.reshape(d_keep, d_keep)

def trace_distance(rho, sigma):
    delta = (rho - sigma)
    delta = (delta + delta.conj().T) / 2.0
    evals = np.linalg.eigvalsh(delta)
    return 0.5 * np.sum(np.abs(evals))

rng = np.random.default_rng(9)

# --- Global fidelity ---
D = 28
H = random_hermitian(D, rng)
E, V = np.linalg.eigh(H)

psi0 = rng.normal(size=(D,)) + 1j * rng.normal(size=(D,))
psi0 /= np.linalg.norm(psi0)
psi0_eig = V.conj().T @ psi0

Tmax = 160.0
N = 700
ts = np.linspace(0.0, Tmax, N)
Fs = np.empty_like(ts)

for i, t in enumerate(ts):
    phase = np.exp(-1j * E * t)
    psit = V @ (phase * psi0_eig)
    Fs[i] = fidelity(psi0, psit)

plt.figure()
plt.plot(ts, Fs, linewidth=1.0)
plt.xlabel("time (arb. units)")
plt.ylabel(r"$F(t)=|\langle\psi(0)|\psi(t)\rangle|^2$")
plt.title("Global fidelity in a finite-D random Hamiltonian (D=28)")
plt.tight_layout()
plt.savefig("ccr_fidelity.png", dpi=150)
plt.close()

# --- Local (coarse-grained) recurrence ---
dims = [2, 2, 2, 2]
Dq = 16
Hq = random_hermitian(Dq, rng)
E_q, V_q = np.linalg.eigh(Hq)

psi0_q = rng.normal(size=(Dq,)) + 1j * rng.normal(size=(Dq,))
psi0_q /= np.linalg.norm(psi0_q)
psi0_q_eig = V_q.conj().T @ psi0_q

rhoA0 = partial_trace_rho(np.outer(psi0_q, psi0_q.conj()), dims, keep=(0,))

Tmax2 = 200.0
N2 = 600
ts2 = np.linspace(0.0, Tmax2, N2)
TDs = np.empty_like(ts2)

for i, t in enumerate(ts2):
    phase = np.exp(-1j * E_q * t)
    psit_q = V_q @ (phase * psi0_q_eig)
    rhoA_t = partial_trace_rho(np.outer(psit_q, psit_q.conj()), dims, keep=(0,))
    TDs[i] = trace_distance(rhoA_t, rhoA0)

plt.figure()
plt.plot(ts2, TDs, linewidth=1.0)
plt.xlabel("time (arb. units)")
plt.ylabel("Trace distance on subsystem A")
plt.title("Local (coarse-grained) recurrence: 4-qubit subsystem A")
plt.tight_layout()
plt.savefig("ccr_trace_distance.png", dpi=150)
plt.close()

print("Plots saved: ccr_fidelity.png, ccr_trace_distance.png")

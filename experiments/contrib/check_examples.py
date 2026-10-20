"""Kiem tra so cho cac vi du trong dong_gop_shape_vector.tex (R = 1, ket qua chia cho pi)."""
import numpy as np
from scipy.integrate import quad

R = 1.0
th0, rho1, rho2 = np.pi / 8, 3 * R / 4, 7 * R / 8
a1 = np.pi / 15
a2 = a1 * (R**2 - rho1**2) / (R**2 - rho2**2)
assert np.isclose(a2, 28 * np.pi / 225)

weights = {
    "uniform": lambda r: 1.0,
    "decay": lambda r: max(0.0, 1 - r / R),
    "step": lambda r: float(r < 13 * R / 16),
}


def int_sector(f, r0, r1, alpha):
    # tich phan cua f(|x|) tren hinh quat vanh khan {r0<=r<=r1, goc rong alpha}
    return alpha * quad(lambda r: f(r) * r, r0, r1, points=[13 * R / 16])[0]


res = {}
for name, f in weights.items():
    U17 = int_sector(f, 2 * R / 3, R, np.pi / 4)
    res[name] = (U17 - int_sector(f, rho1, R, a1), U17 - int_sector(f, rho2, R, a2))
    print(f"{name:8s} w17(O1)/pi = {res[name][0]/np.pi:.6f}  w17(O2)/pi = {res[name][1]/np.pi:.6f}")

# dong dang trong file .tex
assert np.isclose(res["uniform"][0], res["uniform"][1])
assert np.isclose(res["decay"][1] - res["decay"][0], 73 * np.pi / 86400)
assert np.isclose(res["step"][0], 437 * np.pi / 18432)
assert np.isclose(res["step"][1], 497 * np.pi / 18432)
print("ty le khoang cach step/decay =", (5 / 1536) / (73 / 86400))

# tong chu vi phan hoach vong-quat: 2R(pi*Nr + Ntheta)
for Nr, Nt in [(3, 24), (5, 160)]:
    Rk = R * np.arange(Nr + 1) / Nr
    P = sum(Nt * ((2 * np.pi / Nt) * (Rk[k - 1] + Rk[k]) + 2 * (Rk[k] - Rk[k - 1])) for k in range(1, Nr + 1))
    assert np.isclose(P, 2 * R * (np.pi * Nr + Nt)), (Nr, Nt)

# Vi du 3: G = U (2^-k, 2^-k + 4^-k), m(G \ (G+y)) >= K*y voi y = 4^-K, K >= 2
ks = np.arange(1, 40)
a, b = 2.0**-ks, 2.0**-ks + 4.0**-ks
inG = lambda t: np.any((t[:, None] > a) & (t[:, None] < b), axis=1)
pts = np.linspace(0, 0.8, 2_000_001)
for K in range(2, 6):  # luoi buoc 4e-7 du min cho y = 4^-K >= 1e-3
    y = 4.0**-K
    loss = np.mean(inG(pts) & ~inG(pts - y)) * 0.8
    assert loss >= K * y * 0.98, (K, loss, K * y)
print("Tat ca kiem tra deu dat.")

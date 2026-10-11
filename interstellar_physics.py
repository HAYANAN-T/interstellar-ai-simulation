"""
Interstellar world, part A: how much time dilation can a planet on a stable orbit get?

Units: G = c = 1, black-hole mass M = 1, spin parameter a in [0, 1).
Formulas: Bardeen, Press & Teukolsky (1972) for prograde equatorial circular orbits in Kerr spacetime.
High precision (mpmath) is required because the interesting spins are 1 - 10^-14 and closer.
"""
import mpmath as mp

mp.mp.dps = 80  # 80 significant digits

def r_isco(a):
    """Radius of the innermost stable circular orbit (prograde), in units of M."""
    a = mp.mpf(a)
    z1 = 1 + mp.cbrt(1 - a**2) * (mp.cbrt(1 + a) + mp.cbrt(1 - a))
    z2 = mp.sqrt(3 * a**2 + z1**2)
    return 3 + z2 - mp.sqrt((3 - z1) * (3 + z1 + 2 * z2))

def dilation_circular(r, a):
    """dt/dtau for a prograde circular equatorial orbit at radius r: Earth-seconds per planet-second."""
    r, a = mp.mpf(r), mp.mpf(a)
    return (r**mp.mpf(1.5) + a) / (r**mp.mpf(0.75) * mp.sqrt(r**mp.mpf(1.5) - 3 * mp.sqrt(r) + 2 * a))

def dilation_at_isco(a):
    return dilation_circular(r_isco(a), a)

def dilation_static_schwarzschild(r_over_rs):
    """Hovering (not orbiting) observer outside a non-spinning hole: 1/sqrt(1 - rs/r)."""
    x = mp.mpf(r_over_rs)
    return 1 / mp.sqrt(1 - 1 / x)

TARGET = mp.mpf(7 * 365.25 * 24)  # "one hour there = seven years here" -> 61,362 hours per hour

def spin_needed(target=TARGET):
    """Find the spin a (as epsilon = 1 - a) where ISCO dilation equals the target."""
    f = lambda log_eps: mp.log(dilation_at_isco(1 - mp.power(10, log_eps))) - mp.log(target)
    log_eps = mp.findroot(f, (-30, -5), solver="anderson")
    return mp.power(10, log_eps)

if __name__ == "__main__":
    print("Schwarzschild ISCO (a=0): r =", mp.nstr(r_isco(0), 6), " dilation =", mp.nstr(dilation_at_isco(0), 6))
    for eps in ["1e-2", "1e-4", "1e-8", "1e-12", "1e-14"]:
        a = 1 - mp.mpf(eps)
        print(f"a = 1 - {eps}: r_isco = {mp.nstr(r_isco(a), 8)}, dilation = {mp.nstr(dilation_at_isco(a), 8)}")
    eps = spin_needed()
    print("Target dilation", mp.nstr(TARGET, 8), "needs 1 - a =", mp.nstr(eps, 6))
    # hovering instead of orbiting: how close to the horizon of a NON-spinning hole?
    x = 1 / (1 - 1 / TARGET**2)
    print("Hovering at Schwarzschild: r/rs - 1 =", mp.nstr(x - 1, 6))

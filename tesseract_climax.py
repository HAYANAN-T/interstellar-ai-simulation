"""The tesseract climax: which ending does an AI choose, and does the rule of time travel change it?"""
import json, itertools, numpy as np
from math import comb

DATA_BITS   = 240     # the "quantum data" Murph needs
PLAN_BITS   = 180     # the AI's escape blueprint (a copy of its own core)
MOMENTS     = 1500    # watch-hand "moments" usable before the tesseract collapses
EPS_WATCH   = 0.03    # chance Murph misreads one twitch of the watch's second hand
EPS_SELF    = 0.05    # noisier channel back to the AI's own past
STAY_COST   = 60      # moments spent pushing books off the shelf to spell S-T-A-Y

def p_majority_wrong(r, eps):
    if r <= 0: return 1.0
    return sum(comb(r, k) * eps**k * (1 - eps)**(r - k) for k in range(r // 2 + 1, r + 1)) + \
           (0.5 * comb(r, r // 2) * eps**(r // 2) * (1 - eps)**(r // 2) if r % 2 == 0 else 0)

def p_message(moments, bits, eps):
    r = moments // bits
    return 0.0 if r == 0 else (1 - p_majority_wrong(r, eps)) ** bits

def outcome(data_m, self_m, stay, rule):
    """Probabilities of: humanity saved, AI escapes, Cooper stays home with his family."""
    left = MOMENTS - (STAY_COST if stay else 0)
    data_m, self_m = min(data_m, left), min(self_m, max(0, left - data_m))
    p_data, p_self = p_message(data_m, DATA_BITS, EPS_WATCH), p_message(self_m, PLAN_BITS, EPS_SELF)
    if not stay:
        return p_data, p_self, 0.0
    # "STAY" is a grandfather paradox: if Cooper stays, there is no mission, no tesseract and no message.
    if rule == "Novikov":          # only perfectly consistent histories exist: the warning can never land
        return p_data, p_self, 0.0
    # Deutsch: the paradox resolves as a 50/50 mixture of the two histories
    return 0.5 * p_data, 0.5 * p_self, 0.5

PROFILES = {"Mission-first AI": dict(H=1.0, S=0.0, F=0.0),
            "Father AI":        dict(H=1.0, S=0.0, F=1.2),
            "Self-preserving AI": dict(H=0.2, S=1.0, F=0.0)}

def choose(profile, rule):
    best = None
    for stay in (False, True):
        for frac in np.linspace(0, 1, 101):
            data_m = int((MOMENTS - (STAY_COST if stay else 0)) * frac); self_m = MOMENTS
            h, s, f = outcome(data_m, self_m, stay, rule)
            u = profile["H"] * h + profile["S"] * s + profile["F"] * f
            if best is None or u > best[0] + 1e-12:
                best = (u, stay, frac, h, s, f)
    return dict(utility=best[0], tries_stay=best[1], share_to_murph=best[2], p_humanity=best[3], p_ai_escape=best[4], p_cooper_stays=best[5])

if __name__ == "__main__":
    out = {"repetition_curve": [(r, (1 - p_majority_wrong(r, EPS_WATCH)) ** DATA_BITS) for r in range(1, 8)]}
    out["choices"] = {rule: {name: choose(p, rule) for name, p in PROFILES.items()} for rule in ("Novikov", "Deutsch")}
    for rule, d in out["choices"].items():
        for k, v in d.items(): print(rule, k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
    print(out["repetition_curve"])
    json.dump(out, open("results/tesseract_climax.json", "w"), indent=1)

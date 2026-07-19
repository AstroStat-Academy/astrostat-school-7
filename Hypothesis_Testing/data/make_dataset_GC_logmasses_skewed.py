"""Generate a skewed GC log-mass sample for the bootstrap median example.

Writes GC_logmasses_skewed.csv: 90 log10-masses drawn from a skew-normal
distribution, so the sample is clearly asymmetric even in log-scale.
Seeded, so the file is reproducible.
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=314)

n = 120
log_masses = stats.skewnorm.rvs(a=6, loc=4.8, scale=0.7, size=n, random_state=rng)

np.savetxt("GC_logmasses_skewed.csv", log_masses, fmt="%.4f")
print(f"Wrote GC_logmasses_skewed.csv  (n = {n}, "
      f"median = {np.median(log_masses):.3f}, "
      f"skew = {stats.skew(log_masses):.2f})")

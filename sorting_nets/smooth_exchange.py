import numpy as np

def smooth_maximum (x1, x2, t = 1):
  #a = min(x1/t, x2/t)
  #return (a + np.log(np.exp(x1/t - a) + np.exp(x2/t - a) - 1)) * t
  return (np.log(np.exp(x1/t) + np.exp(x2/t) - 1)) * t

def smooth_exchange (x1, r1, x2, r2, t = 1):
  smax = smooth_maximum(x1, x2, t)
  smin = x1 + x2 - smax
  aff1 = (smax - x2) / (x1 - x2)
  aff2 = (x1 - smax) / (x1 - x2)
  rmax = aff1 * r1 + aff2 * r2
  rmin = aff2 * r1 + aff1 * r2
  return smax, rmax, smin, rmin

a, b, c, d = 1, 2, 3, 4
for i in range(10):
  a, b, c, d = smooth_exchange(a, b, c, d, .1)
  print(a, b, c, d)


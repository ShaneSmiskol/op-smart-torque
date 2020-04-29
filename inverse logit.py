import matplotlib.pyplot as plt
import math


def f():
  x = []

  e = math.e
  x0 = 0
  L = 1
  k = 1

  for i in range(-800, 801):
    i = i / 100
    z = L / (1 + e ** (-k * (i - x0)))
    x.append(z)
  return x

x = f()
plt.plot(range(-800, 801), x)

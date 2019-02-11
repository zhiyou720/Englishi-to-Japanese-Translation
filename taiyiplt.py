import matplotlib.pyplot as plt
import numpy as np

readingscore = []
flrascore = []

with open('yaoyi1', 'r') as f1:
    for line in f1.readlines():
        lines = line.strip()
        readingscore.append(lines)

print(readingscore)

with open('taoyi2', 'r') as f1:
    for line in f1.readlines():
        lines = line.strip()
        flrascore.append(lines)

print(flrascore)


plt.figure()
plt.ylim(0,200)
plt.plot(readingscore, flrascore,'o')

plt.show()

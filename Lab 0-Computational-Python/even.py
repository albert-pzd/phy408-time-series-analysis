import numpy as np
import matplotlib.pyplot as plt
from integral import integral

# number of samples
nt = 100
nt2 = 1000

# generate time vector
t = np.linspace(0, np.pi, nt)
t2 = np.linspace(0, np.pi, nt2)

# compute sample interval (evenly sampled, only one number)
dt = t[1] - t[0]
dt2 = t2[1] - t2[0]
y = np.sin(t)
y2 = np.sin(t2)

# plot
plt.figure(1)
plt.plot(t, y, 'r+')
plt.xlabel("t")
plt.ylabel("y")
plt.title("Plot of y(t)")
plt.savefig("Plot of y(t)", dpi = 400)

c = integral(y, dt)
print("The out put of c is:", c)

#Increase the number of samples


t2 = np.linspace(0, np.pi, nt2)
dt2 = t2[1] - t2[0]
y2 = np.sin(t2)

# improved plot
plt.figure(2)
plt.plot(t2, y2)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Plot of y(t) with improved accuracy")
plt.savefig('Improved Plot', dpi = 400)

c2 = integral(y2, dt2)
print("The improved result is:", c2)
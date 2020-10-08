import numpy as np
import matplotlib.pyplot as plt
from integral import integral

nt_values = [20, 50, 100, 500, 1000]
c_values = [0, 0, 0, 0, 0]
number = 1

#for loop
for index in range(len(nt_values)):
    nt = nt_values[index]
    # sampling between [0,0.5]
    t1 = np.linspace(0, 0.5, nt)
    # double sampling between [0.5,1]
    t2 = np.linspace(0.5, 1, 2*nt)
    # concatenate time vector
    t = np.concatenate((t1[:-1], t2))
    # compute y values 
    y = np.sin(2 * np.pi * (8*t - 4*t**2))
    
    #plot for y(t)
    plt.figure(number)
    plt.plot(t, y)
    plt.xlabel("Time t")
    plt.ylabel("y")
    plt.title("Plot of y(t) for nt=" + str(nt_values[index]))
    plt.savefig("Plot of y(t) for nt=" + str(nt_values[index]), dpi = 400)
    number += 1
    
    # compute sampling interval vector
    dt = t[1:] - t[:-1]
    c = integral(y, dt)
    c_values[index] = c
    print("The result of c is :", c)
    
    
plt.figure(number)
plt.plot(nt_values, c_values, 'o', label = 'Data Points')
plt.plot(nt_values, c_values, ':')
plt.xlabel("Value of nt")
plt.ylabel("Value of c")
plt.title("Plot of c(nt)")
plt.legend()
plt.savefig("Plot of c(nt)", dpi= 400, bbox_inches='tight')
number += 1

frequency = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
dt = 0.0005

pic = 0
for f in frequency:
    t = np.arange(0, 10, 0.5)
    t2 = np.arange(0, 10, dt)
    g = np.cos(2 * np.pi * f * t)
    g2 = np.cos(2 * np.pi * f * t2)
    c = integral(g, dt)
    plt.figure(number)
    plt.plot(t2, g2, ':')
    plt.plot(t, g,'x', color='red')
    plt.title("Sampling Time Series Plot Frequency = " + str(f) + 'Hz')
    plt.xlabel("Time t")
    plt.ylabel("Value of g(t)")
    plt.savefig("Plot of Frequency {}".format(pic), dpi=400)
    number += 1
    pic += 1
    
import numpy as np
import matplotlib.pyplot as plt
import time
#load data & define constants
time_full, displacement = np.loadtxt('RAYN.II.LHZ.txt', unpack = True)
time_800 = time_full[time_full <= 800]
time_800 = time_800[time_800 >= 0]
displacement_800 = displacement[:len(time_800)]
t_H = np.array([10, 20])



#Convolution function
def myConv(f, w):
    result = np.zeros(len(f)+len(w)-1)
    for i1 in range(len(result)):
        for i2 in range(len(f)):
            if i1-i2>=0 and i1-i2< len(w):
                result[i1] = result[i1] + f[i2] * w[i1-i2]
    return result

#H,D function
def HD(n, dt):
    H= np.ones(n)
    D = np.zeros(n)
    H[0] = 0.5
    D[0] = 1 / dt
    return H,D

#RLresponse function
dt = 0.0002
t_range = np.arange(0, 0.1, dt)
n = len(t_range)
def RLresponse(R, L, V_in, dt):
    R_t = HD(n, dt)[1] - (R/L)*np.e**(-R*t_range/L)*HD(n, dt)[0]
    return np.convolve(V_in, R_t)*dt

#Gaussian function
def g1(t):
    return np.e**(-(t/t_H[0])**2) / (np.sqrt(np.pi)*t_H[0])

def g2(t):
    return np.e**(-(t/t_H[1])**2) / (np.sqrt(np.pi)*t_H[1])


#Part 1 Q3
f = np.random.random([50])
w = np.random.random([100])
np_result = np.convolve(f, w)
my_result = myConv(f, w)
x_axis = np.arange(len(my_result))
print('The first 5 output from myConv:', my_result[0], my_result[1], 
      my_result[2], my_result[3], my_result[4])
print('The first 5 output from Numpy:', np_result[0], np_result[1], 
      np_result[2], np_result[3], np_result[4])


plt.figure(0)
plt.plot(x_axis, my_result, label = 'Result from myConv')
plt.plot(x_axis, np_result, label = 'Result from Numpy')
plt.xlabel('Ouput array indices')
plt.ylabel('Convoluted Results')
plt.legend()
plt.title('Ouput differnece: myConv vs numpy.convolve')
plt.savefig('p1q3', dpi=400)

#Part 1 Q4
cases = [10,100,200,400,600,800,1000]
my_time = np.zeros([7])
np_time = np.zeros([7]) 
for index in range(7):
    f2 = np.random.random([cases[index]])
    w2 = np.random.random([cases[index]])
    t1 = time.time()
    g_my = myConv(f2, w2)
    t2 = time.time()
    g_np = np.convolve(f2, w2)
    t3 = time.time()
    np_time[index] = t3 - t2
    my_time[index] = t2 - t1
    
#Comparison plot
plt.figure(1)
plt.plot(cases, my_time, label = 'Result from myConv')
plt.plot(cases, np_time, label = 'Result from Numpy')
plt.xlabel('Size')
plt.ylabel('Run-time')
plt.legend()
plt.title('Speed Comparison: myConv vs numpy.convolve') 
plt.savefig('p1q4', dpi=400)

#Part 2 Q3
R = 850
L = 2
theoretical_R = HD(n, dt)[1] - (R/L)*np.exp(-R*t_range/L)*HD(n, dt)[0]
theoretical_S = np.e**(-R*t_range/L)*HD(n, dt)[0]
V1, V2 = HD(n, dt)
V_impulse = RLresponse(R, L, V2, dt)
V_impulse = V_impulse[:n]
V_step = RLresponse(R, L, V1, dt)
V_step = V_step[:n]

plt.figure(3)
plt.plot(t_range, V_impulse, ls='',marker='o', label = 'Output time series(convoluted)')
plt.plot(t_range, theoretical_R, label = 'Theoretical output')
plt.xlabel('Range of t')
plt.ylabel('Convoluted Result')
plt.title('Impulse Response: Theoretical ouput vs Convoluted output')
plt.legend()
plt.savefig('p2q31', dpi = 400, bbox_inches='tight')

plt.figure(4)
plt.plot(t_range, V_step, ls='',marker='o', label = 'Output time series(convoluted)')
plt.plot(t_range, theoretical_S, label = 'Theoretical output')
plt.xlabel('Range of t')
plt.ylabel('Convoluted Result')
plt.title('Step Response: Theoretical ouput vs Convoluted output')
plt.legend()
plt.savefig('p2q32', dpi = 400, bbox_inches='tight')

#Part 3 Q1
p3_dt = time_full[1] - time_full[0]
time_1 =  np.arange(-30, 30, p3_dt)
time_2 = np.arange(-60, 60, p3_dt) 

plt.figure(5)
plt.plot(time_800, displacement_800)
plt.xlabel('Time(t)')
plt.ylabel('Displacement(m)')
plt.title('Raw Synthetic Seismogram: RAYN 0 to 800')
plt.savefig('Synthetic Seismogram', dpi = 400, bbox_inches='tight')

plt.figure(6)
plt.plot(time_1, g1(time_1), label = 't_H=10')
plt.plot(time_2, g2(time_2), label = 't_H=20')
plt.xlabel('Time(t)')
plt.ylabel('Gaussian g(t)')
plt.title('Gaussian functions')
plt.legend()
plt.savefig('p3q12', dpi=400)

#Part 3 Q2
convoluted_10 = np.convolve(g1(time_1), displacement_800, mode='same') * p3_dt
convoluted_20 = np.convolve(g2(time_2), displacement_800, mode='same') * p3_dt

plt.figure(7)
plt.plot(time_800, convoluted_10, label = 't_H=10')
plt.plot(time_800, convoluted_20, label = 't_H=20')
plt.plot(time_800, displacement_800, label = 'Raw data')
plt.xlabel('Time(t)')
plt.ylabel('Displacement(m)')
plt.title('Convolved time series versus Raw data')
plt.savefig('Convolved time series versus Raw data', dpi=400, bbox_inches='tight')
plt.legend()

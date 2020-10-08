import numpy as np
import matplotlib.pyplot as plt
from co2data import co2Data, time

"""Part 1"""
#setting constants
f_s = 12
f_0 = 1
M = 1.04
epsilon = 0.04
delta = 1.0/f_s
q = np.exp(-1j*2*np.pi*f_0/f_s)
p = (1+epsilon)*q
f_range = np.linspace(-f_s/2, f_s/2, 1000)

def W(f):
    z = np.exp(-1j*2*np.pi*f*delta)
    q_c = np.conj(q)
    p_c = np.conj(p)
    return M * ((z-q)/(z-p)) * ((z-q_c)/(z-p_c))

w_f = W(f_range)

plt.figure()
plt.plot(f_range, w_f * np.conj(w_f)) 
plt.xlabel("Cycles per year(f)")
plt.ylabel("|W(f)|^2")
plt.xlim(-1.5, 1.5)
plt.ylim(0.5, 1.0)
plt.title("|W(f)|^2 Power Spectrum")
plt.savefig("P1Q3", dpi=400)


"""Part 2"""
a, b, c = [0.962, -1.665, 0.962]
B, C = [-1.665, 0.925]
dt = 1 / f_s
year_range = np.arange(0, 100, dt)

def ratFilter(N, D, x):
    first_filter = np.convolve(N, x)[:-2]
    result = np.zeros(len(x))
    result[0] = 1
    for i in range(len(x)):
        D_values = []
        for i2 in range(1,3):
            if i-i2 >= 0:
                D_values.append(D[i2]*result[i-i2])
        result[i] = (first_filter[i] - sum(D_values)) / D[0]
    return result

N = [a,b,c]
D = [1,B,C]

delta = np.zeros(1200)
delta[0] = dt
result = ratFilter(N, D, delta)

plt.figure()
plt.plot(year_range[0:72], result[0:72]) # 1200 samples for 100 years, so first 72 samples are for the first 6 years. 
plt.xlabel("Year")
plt.ylabel("Impulse Response")
plt.title("Impulse Response from 0 to 6 years")
plt.savefig("P2Q3", dpi = 400)

f_result = np.fft.fft(result)
plt.figure()
plt.plot(year_range[0:72], f_result[0:72]) # 1200 samples for 100 years, so first 72 samples are for the first 6 years. 
plt.xlabel("Year")
plt.ylabel("Impulse Response")
plt.title("FT Impulse Response from 0 to 6 years")
plt.savefig("P2Q32", dpi = 400)



"""Part 3"""

dt2 = time[1] - time[0]
coef = np.polyfit(time, co2Data, 1)
detrend_co2  = co2Data - np.polyval(coef, time)
trend = np.polyval(coef, time)

plt.figure()
plt.plot(time, co2Data, '.', label = 'Data Points')
plt.plot(time, np.polyval(coef, time), label = 'Fitted Line')
plt.xlabel("Time")
plt.ylabel("CO2 Concentration")
plt.title("Original Data with Fitted line")
plt.legend()
plt.savefig("P3Q1", dpi = 400)

plt.figure()
plt.plot(time, co2Data, label = 'Original Data')
plt.plot(time, detrend_co2, label = 'Detrended Data')
plt.xlabel("Time")
plt.ylabel("CO2 Concentration")
plt.title("Original data and Detrended data")
plt.legend()
plt.savefig("P3Q12", dpi=400)

filtered_co2 = ratFilter(N,D, detrend_co2)
filtered_co2 = filtered_co2 + trend

fft_co2 = np.fft.fftshift(np.fft.fft(detrend_co2))
freq_co2 = np.fft.fftshift(np.fft.fftfreq(len(detrend_co2), dt2))

plt.figure()
plt.plot(freq_co2, np.abs(fft_co2))
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Amplitude Spectrum")
plt.savefig("P3Q31",dpi = 400)

plt.figure()
plt.plot(freq_co2, np.angle(fft_co2))
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.title("Phase Spectrum")
plt.savefig("P3Q32",dpi = 400)

plt.figure()
plt.plot(freq_co2, np.abs(fft_co2))
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Amplitude Spectrum")
plt.xlim(0, 3.5)
plt.savefig("P3Q33",dpi = 400)

plt.figure()
plt.plot(freq_co2, np.angle(fft_co2))
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.title("Phase Spectrum")
plt.xlim(0, 3.5)
plt.savefig("P3Q34",dpi = 400)

for i in range(len(fft_co2)):
    if abs(freq_co2[i]) > 0.9:
        fft_co2[i] = 0

transformed = np.fft.ifft(fft_co2) + trend

plt.figure()
plt.plot(time, co2Data, label = "Original Data")
plt.plot(time, filtered_co2, label = "Notch Filtered Data")
plt.plot(time, transformed, label = "f-domain Filtered Data")
plt.title("Different methods of Filtering")
plt.legend()
plt.xlabel("Time")
plt.ylabel("CO2 concentration")
plt.savefig("P3Q4", dpi = 400)

original_co2 = ratFilter(N,D,co2Data)
fft_original = np.fft.fft(original_co2)

for i in range(len(fft_original)):
    if abs(freq_co2[i]) > 0.9:
        fft_original[i] = 0
        
transformed_original = np.fft.ifft(np.real(fft_original))

plt.figure()
plt.plot(time, co2Data, label = "Original Data")
plt.plot(time, original_co2, label = "Notch Filtered Data")
plt.plot(time, transformed_original, label = "f-domain Filtered Data")
plt.title("Applying Filters Without Detrending")
plt.legend()
plt.xlabel("Time")
plt.ylabel("CO2 concentration")
plt.savefig("P3Q51", dpi = 400)
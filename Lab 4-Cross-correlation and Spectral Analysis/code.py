import matplotlib.pyplot as plt
import numpy as np

"""load data"""
mlac = np.genfromtxt("MLAC_data.txt").flatten()
phl = np.genfromtxt("PHL_data.txt").flatten()
nwao = np.genfromtxt('nwao.vh1')

"""Question 1"""

correlation = np.fft.fftshift(np.fft.ifft(np.conj(np.fft.fft(phl))*np.fft.fft(mlac))) 
tau = np.linspace(-len(correlation)/2, len(correlation)/2, len(correlation))

plt.figure()
plt.plot(tau, correlation)
plt.xlim(-260,260)
plt.xlabel(r'$\tau$')
plt.ylabel("Correlation")
plt.title("Cross-correlation Plot Zoomed In")
plt.savefig("Q1P1", dpi = 400, bbox_inches='tight')

bit_mlac = np.sign(mlac)
bit_phl = np.sign(phl)

bit_correlation = np.fft.fftshift(np.fft.ifft(np.conj(np.fft.fft(bit_phl))*np.fft.fft(bit_mlac)))

plt.figure()
plt.plot(tau, correlation, label = "True Correlation")
plt.plot(tau, bit_correlation / 15e12, label = "Bit Converted (Scaled)")
plt.xlim(-260,260)
plt.xlabel(r'$\tau$')
plt.ylabel("Correlation")
plt.title("Bit Conversion Zoomed at [-260, 260]")
plt.legend()
plt.savefig("Q1P2", dpi = 400, bbox_inches='tight')

plt.figure()
plt.plot(tau, correlation, label = "True Correlation")
plt.plot(tau, bit_correlation / 15e12, label = "Bit Converted (Scaled)")
plt.xlim(0,200)
plt.xlabel(r'$\tau$')
plt.ylabel("Correlation")
plt.title("Bit Conversion Zoomed at [0, 200]")
plt.legend()
plt.savefig("Q1P3", dpi = 400, bbox_inches='tight')

"""Question 2"""

dt = 10
M = 9
time = nwao[:,0]
velocity = nwao[:,1]

plt.figure()
plt.plot(time / 3600, velocity)
plt.xlabel("Time(h)")
plt.ylabel("Velocity(counts)")
plt.title("Seismic Raw Data NWAO")
plt.savefig("Q2P1",dpi = 400, bbox_inches='tight')

velocity_fft = np.fft.fftshift(np.fft.fft(velocity))
power_spectrum = np.abs(velocity_fft) ** 2
freq = np.fft.fftshift(np.fft.fftfreq(len(time), dt)) * 1000

plt.figure()
plt.plot(freq, power_spectrum)
plt.xlabel("Frequency(mHz)")
plt.ylabel("Amplitude")
plt.title("Power Spectrum")
plt.xlim(0,50)
plt.savefig("Q2P2", dpi = 400, bbox_inches='tight')

coef = np.polyfit(time, velocity, 1)
trend = np.polyval(coef, time)

velocity_detrend = velocity - trend
w_n = 1 - np.cos(2*np.pi*np.arange(len(time))/len(time)) 
velocity_hanning = velocity_detrend * w_n
hanning_fft = np.abs(np.fft.fftshift(np.fft.fft(velocity_hanning)))**2

plt.figure()
plt.plot(freq, hanning_fft)
plt.xlabel("Frequency(mHz)")
plt.ylabel("Amplitude")
plt.title("Power Spectrum with Hanning Window")
plt.xlim(0,50)
plt.ylim(1,8e13)
plt.savefig("Q2P3", dpi = 400, bbox_inches='tight')

plt.figure()
plt.plot(freq, hanning_fft, label = "Processed Data")
plt.plot(freq, power_spectrum, label = "Original Data")
plt.xlabel("Frequency(mHz)")
plt.ylabel("Amplitude")
plt.title("Power Spectrum with Hanning Window")
plt.xlim(0.1,2.6)
plt.ylim(0,6e14)
plt.legend()
plt.savefig("Q2P4", dpi = 400, bbox_inches='tight')

plt.figure()
plt.plot(freq, hanning_fft, label = "Processed Data")
plt.plot(freq, power_spectrum, label = "Original Data")
plt.xlabel("Frequency(mHz)")
plt.ylabel("Amplitude")
plt.title("Power Spectrum 0.1-1.2 mHz")
plt.xlim(0.1,1.2)
plt.ylim(0,4e13)
plt.legend()
plt.savefig("Q2P42", dpi = 400, bbox_inches='tight')

plt.figure()
plt.plot(freq, hanning_fft, label = "Processed Data")
plt.plot(freq, power_spectrum, label = "Original Data")
plt.xlabel("Frequency(mHz)")
plt.ylabel("Amplitude")
plt.title("Power Spectrum 1.2-2.6 mHz")
plt.xlim(1.2,2.6)
plt.ylim(0,6e14)
plt.legend()
plt.savefig("Q2P43", dpi = 400, bbox_inches='tight')

plt.figure(figsize=(12,6))
plt.plot(freq, hanning_fft)
plt.xlabel("Frequency(mHz)")
plt.ylabel("Amplitude")
plt.title("Power Spectrum 1.2-2.6 mHz Annotated")
plt.xlim(0.1,2.6)
plt.ylim(0,8e13)
plt.annotate('1',xy=(0.66, 1e13))
plt.annotate("2",xy=(0.75, 1.15e13))
plt.annotate("3",xy=(0.92, 3.1e13)) 
plt.annotate("4",xy=(1.02, 0.85e13)) 
plt.annotate("5",xy=(1.06, 0.85e13)) 
plt.annotate("6",xy=(1.09, 0.7e13))
plt.annotate("7",xy=(1.15, 0.85e13)) 
plt.annotate("8",xy=(1.21, 4.35e13)) 
plt.annotate("9",xy=(1.34, 1.7e13)) 
plt.annotate("10",xy=(1.36, 1.65e13)) 
plt.annotate("11",xy=(1.40, 1.62e13)) 
plt.annotate("12",xy=(1.46, 3.3e13)) 
plt.annotate("13",xy=(1.55, 5.4e13)) 
plt.annotate("14",xy=(1.60, 5.0e13)) 
plt.annotate("15",xy=(1.66, 1.9e13)) 
plt.annotate("16",xy=(1.72, 3.9e13)) 
plt.annotate("17",xy=(1.85, 4.1e13))
plt.annotate("18",xy=(1.97, 3.5e13)) 
plt.annotate("19",xy=(2.08, 6.3e13)) 
plt.annotate("20",xy=(2.22, 7.1e13)) 
plt.annotate("21",xy=(2.33, 2e13)) 
plt.annotate("22",xy=(2.42, 6.1e13)) 
plt.annotate("23",xy=(2.55, 4.3e13)) 
plt.savefig("Q2P5", dpi = 400, bbox_inches='tight')



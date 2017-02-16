import pyaudio
import numpy as np
# enter the required frequency resolution that you would like
reqdRes = 30 #Hz
fs = 44100 # Hz 
CHANNELS = 2
WIDTH = 2 
# required resolution = fs/N, where N  is the size of the fft
# find the closest power of 2 greater than fs/reqdRes 
fftLen = 2**(np.ceil(np.log2(fs/float(reqdRes))))
CHUNK = 4096 #number of samples processed at a time 

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
	        channels=CHANNELS,
	        rate=fs,
	        input=True,
	        output=True,
	        frames_per_buffer=CHUNK,
		)
# The arrays to plot frequency and time
R = []
# power thresold for deciding noise or music
threshold = 1.0
# frequency array 
freq = np.linspace(0,int(fs),fftLen/2)
background = 1000.0
while(1):	
	data = stream.read(CHUNK)
	input_wave = np.fromstring(data, dtype=np.int16)
# take FFT of newest frame 
	R = np.abs(np.fft.fft(input_wave[0:fftLen]))[0:fftLen/2]/np.sqrt(fftLen)
# find average power in the frame	
	pwr = abs((1/float(fftLen/2.0))*np.sum(np.square(input_wave[0:fftLen/2.0])))
# find the power of the frame with minimum power while recording (assumption - minimum power would be that of the background noise)	
# and assign that to the variable 'background'
	if pwr<background:
		background = pwr
# find the ratio of power of the current frame (pwr) to the minimum power recorded till now (background) 	
# in order to have a relative idea of how loud the input is. rel_pwr will always be >1 as if pwr is less than background (i.e. the power 
# of the current frame is less than the minimum), then the minimum power (background) will be updated.
	rel_pwr = pwr/float(background)
# find the peak frequency
	funda_freq = abs(freq[np.argmax(R)])
	funda_freq = funda_freq if funda_freq<fs/2 else fs-funda_freq
# decide whether noise, music or voice based on relative power and peak frequency
# threshold is manually decided by checking the rel_pwr variable as some noise is brought near the microphone
	if rel_pwr>threshold and funda_freq > 370:
		print 'music'
		print funda_freq
	elif rel_pwr<=threshold:
		print 'noise'
	elif rel_pwr>threshold and funda_freq <= 370:
		print 'voice'
		print funda_freq
	print ' '
stream.stop_stream()
stream.close()
p.terminate() 


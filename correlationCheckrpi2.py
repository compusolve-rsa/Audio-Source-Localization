# Program to listen to audio and move the roomba towards the source of audio
import pyaudio
import numpy as np
import struct
import serial
import time	
from moveRoomba import roomba_start, roomba_rotate, roomba_move, roomba_stop
from spectralFeatures_functions import elliptic_filterit, fftWindowFilter,stft,spec_plot
import matplotlib.pyplot as plt
port = serial.Serial("/dev/ttyAMA0",baudrate = 115200, timeout =3.0)
from scipy.signal import butter,lfilter, freqz, ellip
block = 4096 #number of samples processed at a time 
fs = 44100 # Hz 
CHANNELS = 2
WIDTH = 2 
sound_speed = 340.29*100 # cm/s
L = block
N = 4000
M = N/2
freq = np.linspace(0,fs,block)
t = np.linspace(0,float(block)/fs, np.ceil(float(block)/M)+1)	
t1 = np.linspace(0,float(block/2)/fs, np.ceil(float(block/2)/M)+1)	
print 'len(t)',len(t),'len(freq)',len(freq)

# start up the roomba
roomba_start(port)

# open pyaudio object
p = pyaudio.PyAudio()

# power thresold 
threshold = 1.7
background = 10000.0

# angle coordinates : (top view of the roomba)
#
#		       ##|##    
#  top left	    #    |    #      top right
#		  # +ve  |  -ve #     
#		 #       |       #   
#		 #---------------#      
#		 #       |       #   
# bottom left	  #  +ve |  -ve #   bottom right
# 		    #    |    # 
# 		       ##|##
#
# The roomba listens for a sound, find the angle of the sound
# then rotates counter-clockwise again by a small angle, listens again and finds the angle
# if the angle reduced (less positive or more negative), the sound source was in front.
# if the angle increased (more positive or less negative), the sound source was at the back.	
# if the object is at top left of the roomba, it rotates by a positive angle theta (counter-clockwise)
# if the object is at top right of the roomba, it rotates by a negative angle theta (clockwise) 
# if the object is at bottom left of the roomba, it rotates by theta+90 (counter-clockwise) , theta being positive 	 
# if the object is at bottom right of roomba, it rotates by theta-90 (clockwise), theta being negative

def find_angle(leftch,rightch):	
# 	normalized left and right channels (interleaved in the signal)	
	
	zcr= ((signal[:-1] * signal[1:]) < 0).sum()
	
#	leftch = filterit(leftch,320,'low',fs)
#	rightch = filterit(rightch,320,'low',fs)	
	print zcr	
# 	find time difference between the two signals reaching the two microphones
# 	using cross-correlation	 	
	corr_arr = np.correlate(leftch,rightch,'full')	
	max_index = (len(corr_arr)/2)-np.argmax(corr_arr)  
	print 'max_index',max_index
	time_d = max_index/float(fs)
#	print 'time diff',time_d
	val = (time_d*sound_speed)/(22.0-8.0) # update distance between microphones	
	print 'val',val

# 	if it's valid then move
	if (val<1 and val>-1) and time_d !=0 and zcr>=2000:
		print 'case 1'	
		angle = 90.0-(180.0/3.14)*np.arccos(val) # distance between microphones = 29 cm
		print angle,'degrees'		
#		plt.plot(corr_arr)
#		plt.show()
	elif (val>1 or val<-1 or time_d==0 or zcr< 2000):
	# invalid value : dont move
		print 'something''s wrong'
		angle = 0.0
	print angle,'degrees'		
	return angle	
b, a = ellip(4, 5, 60, [65.0/(fs/2),350.0/(fs/2)], 'band', analog=False)
#b,a = butter(20, 500.0/(fs/2.0), btype='low', analog=False, output='ba')
#w, h = freqz(b, a)
#plt.semilogx(w, 20 * np.log10(abs(h)))
#plt.title('butter filter frequency response (rp=5, rs=60)')
#plt.xlabel('Frequency [radians / second]')
#plt.ylabel('Amplitude [dB]')
#plt.margins(0, 0.1)
#plt.grid(which='both', axis='both')
#plt.axvline(100, color='green') # cutoff frequency
#plt.axhline(-40, color='green') # rs
#plt.axhline(-5, color='green') # rp
#plt.show()

while(1):	
#	start stream
	stream = p.open(format=p.get_format_from_width(WIDTH),
	        channels=CHANNELS,
	        rate=fs,
	        input=True,
	        output=True,
	        frames_per_buffer=block,
		)	
# 	read data	
	data = stream.read(block)
	# once the data has been acquired, stop listening
	stream.stop_stream()
	stream.close()
	shorts = (struct.unpack( 'h' *2* block, data ))
	signal=np.array(list(shorts),dtype=float)
#	X = stft(signal,L, N, M)
#	spec_plot(freq,t,X,'Frequency(Hz)','Time(s)','Spectrogram before filtering') 
#	noisepwr = np.sum(np.square(signal))/float(len(signal))
# 	divide into left and right streams	
	leftch = signal[0::2]
	rightch= signal[1::2] 
#	X1 = stft(leftch,L, N, M)
#	spec_plot(freq,t1,X1,'Frequency(Hz)','Time(s)','Spectrogram after filtering')
	"""
	fft_signal  = abs(np.fft.fft(leftch, 8000))
	fft_lsignal = abs(np.fft.fft(rightch,8000)) 
	fft_rsignal = abs(np.fft.fft(signal ,8000)) 
	
	fft_pwr  = np.sum(np.square(fft_signal ))/float(len(fft_lsignal)+len(fft_rsignal))
	fft_lpwr = np.sum(np.square(fft_lsignal))/float(len(fft_lsignal)+len(fft_rsignal))
	fft_rpwr = np.sum(np.square(fft_rsignal))/float(len(fft_lsignal)+len(fft_rsignal))

	plt.plot(fft_signal, 'g')
	plt.plot(fft_lsignal,'b')
	plt.plot(fft_rsignal,'r')
	plt.show()
	
	print 'fft power',fft_pwr,(fft_lpwr+fft_rpwr)/2.0	
	"""
# find the power in the right channel and left channel separately, and in the overall stream
#	lnoisepwr = np.sum(np.square(leftch))/float(len(leftch)+len(rightch))
#	rnoisepwr = np.sum(np.square(rightch))/float(len(rightch)+len(leftch))
#	print 'before',noisepwr, lnoisepwr+rnoisepwr
# verify that they're equal

# 	now filter the left stream and the right stream with the same filter
	filtered_lnoise = lfilter(b,a,leftch)
	filtered_rnoise = lfilter(b,a,rightch)
#	filtered_lnoise =  elliptic_filterit(leftch,np.array([1000,3000]),4,60,fs)
#	filtered_rnoise = elliptic_filterit(rightch,np.array([1000,3000]),4,60,fs)
#	filtered_rnoise = elliptic_filterit(rightch,np.array([705,2000]),4,60,fs)
# 	view the filtered output in the fft domain!
#	filtered_lnoise = fftWindowFilter(leftch,np.array([1000,3000]), 'band',fs)
#	filtered_rnoise = fftWindowFilter(rightch,np.array([1000,3000]), 'band',fs)
	"""	
	filt_fft_signal  = abs(np.fft.fft(filtered_lnoise, 8000))
	filt_fft_lsignal = abs(np.fft.fft(filtered_rnoise,8000)) 
	filt_fft_rsignal = abs(np.fft.fft(filtered_noise ,8000)) 
	
	filt_fft_pwr  = np.sum(np.square(filt_fft_signal ))/float(len(filt_fft_lsignal)+len(filt_fft_rsignal))
	filt_fft_lpwr = np.sum(np.square(filt_fft_lsignal))/float(len(filt_fft_lsignal)+len(filt_fft_rsignal))
	filt_fft_rpwr = np.sum(np.square(filt_fft_rsignal))/float(len(filt_fft_lsignal)+len(filt_fft_rsignal))

	plt.plot(filt_fft_signal, 'g')
	plt.plot(filt_fft_lsignal,'b')
	plt.plot(filt_fft_rsignal,'r')
	plt.show()
	"""
	lfnoisepwr = np.sum(np.square(filtered_lnoise))/float(len(filtered_lnoise)+len(filtered_rnoise))
	rfnoisepwr = np.sum(np.square(filtered_rnoise))/float(len(filtered_lnoise)+len(filtered_rnoise))

#	fnoisepwr = np.sum(np.square(filtered_noise))/float(len(filtered_noise))
#	pwr = fnoisepwr
#	print lfnoisepwr+rfnoisepwr, fnoisepwr
#	print lnoisepwr+rnoisepwr, noisepwr
#	pwr = np.sum(np.square(filt_signal))/float(len(filt_signal))	

	pwr = lfnoisepwr+rfnoisepwr
#	print pwr, fnoisepwr
#	pwr = lnoisepwr+rnoisepwr
	if background > pwr:
		background = pwr
#	print 'background',background
	pwr_back = pwr/float(background)	
	if pwr_back>threshold:
		print 'listening...'
		angle1 = find_angle(filtered_lnoise,filtered_rnoise)

#		if the angle is sufficiently large, rotate counter-clockwise by 5 degrees and listen again, 
		if abs(angle1)>1:
	#		now rotate counter-clockwise by  5 degrees and listen again
			print 'preliminary rotation...'
			roomba_rotate(port, 10, 150)
			time.sleep(0.2)
			print 'listening again...'
	#		once again open the audio stream and listen for another block of data	
			stream = p.open(format=p.get_format_from_width(WIDTH),
				channels=CHANNELS,
				rate=fs,
				input=True,
				output=True,
				frames_per_buffer=block,
				)	
			data = stream.read(block)
			# stop listening again
			stream.stop_stream()
			stream.close()	
	
			shorts = (struct.unpack( 'h' *2* block, data ))
			signal=np.array(list(shorts),dtype=float)
			leftch = signal[0::2]
			rightch= signal[1::2] 
#			filtered_lnoise =  elliptic_filterit(leftch,np.array([1000,3000]),4,60,fs)
#			filtered_rnoise = elliptic_filterit(rightch,np.array([1000,3000]),4,60,fs)
			filtered_lnoise = lfilter(b,a,leftch)
			filtered_rnoise = lfilter(b,a,rightch)
#			filtered_lnoise = fftWindowFilter(leftch,np.array([1000,3000]), 'band',fs)
#			filtered_rnoise = fftWindowFilter(rightch,np.array([1000,3000]), 'band',fs)
#			new angle found		
# 			if the angle has 'reduced' (reduced on the absolute scale i.e. less positive for a positive angle or more negative for negative angle)			
#			then the source was in front. If the angle increased, the source was at the back. 			
			angle2 = find_angle(filtered_lnoise,filtered_rnoise)
			sgn = 1 if angle1>0 else -1 # check if object was initially in left or right hemisphere

			if angle2<angle1:		
				print 'source is in front of me...' 	
				roomba_rotate(port,int(angle2),200)
				time.sleep(0.2)	
				roomba_move(port,150,300)
				time.sleep(0.2) 						
			else:
				print 'source is behind me...'
				roomba_rotate(port,int((sgn*90)+angle2),200)
				time.sleep(0.2)	
				roomba_move(port,150,300)
	
#				if 	
	
			print 'done rotating...'
		else: 
#		angle wasn't large enough	
			print 'angle wasn''t large enough to rotate:',angle1
	else:	
#	sound wasn't loud enough - do nothing	
		stream.stop_stream()
		stream.close()
		print 'no sound detected'
	print ' '
	



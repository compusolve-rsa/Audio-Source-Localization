# Program to listen to audio and move the roomba towards the source of audio
import pyaudio
import numpy as np
import struct
import serial
import time	

# separate file with functions written to move the Roomba
from moveRoomba import roomba_start, roomba_rotate, roomba_move, roomba_stop

import matplotlib.pyplot as plt

# open the serial port
port = serial.Serial("/dev/ttyAMA0",baudrate = 115200, timeout =3.0)

# import filters 
from scipy.signal import butter,lfilter, freqz, ellip
block = 4096 #number of samples processed at a time 
fs = 44100 # sampling frequency in Hz 
CHANNELS = 2
WIDTH = 2 
sound_speed = 340.29*100 # cm/s

# define some parameters for the STFT
L = block  # size of a block
N = 4000  #  
M = N/2   # Hopsize
freq = np.linspace(0,fs,block)
t = np.linspace(0,float(block)/fs, np.ceil(float(block)/M)+1)	
t1 = np.linspace(0,float(block/2)/fs, np.ceil(float(block/2)/M)+1)	
#print 'len(t)',len(t),'len(freq)',len(freq)

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
	
# 	find time difference between the two signals reaching the two microphones
# 	using cross-correlation	 	
	corr_arr = np.correlate(leftch,rightch,'full')	
	max_index = (len(corr_arr)/2)-np.argmax(corr_arr)  
	print 'max_index',max_index
	time_d = max_index/float(fs)

	val = (time_d*sound_speed)/(22.0-8.0) # update distance between microphones	
	print 'val',val

# 	if it's valid then move
	if (val<1 and val>-1) and time_d !=0 and zcr>=2000:
		print 'case 1'	
		angle = 90.0-(180.0/3.14)*np.arccos(val) # distance between microphones = 29 cm
		print angle,'degrees'		
	elif (val>1 or val<-1 or time_d==0 or zcr< 2000):
	# invalid value : dont move
		print 'something''s wrong'
		angle = 0.0
	print angle,'degrees'		
	return angle	

# return the filter coefficients for the elliptic filter
b, a = ellip(4, 5, 60, [65.0/(fs/2),350.0/(fs/2)], 'band', analog=False)

while(1):	
#	start PyAudio stream
	stream = p.open(format=p.get_format_from_width(WIDTH),
	        channels=CHANNELS,
	        rate=fs,
	        input=True,
	        output=True,
	        frames_per_buffer=block,
		)	
# 	read a block of data	
	data = stream.read(block)
	# once the data has been acquired, stop listening
	stream.stop_stream()
	stream.close()
	shorts = (struct.unpack( 'h' *2* block, data ))
	signal=np.array(list(shorts),dtype=float)
# 	divide into left and right streams	
	leftch = signal[0::2]
	rightch= signal[1::2] 

# 	now filter the left stream and the right stream with the same filter
	filtered_lnoise = lfilter(b,a,leftch)
	filtered_rnoise = lfilter(b,a,rightch)
# 	find the power in the left and right channels	
	lfpwr = np.sum(np.square(filtered_lnoise))/float(len(filtered_lnoise)+len(filtered_rnoise))
	rfpwr = np.sum(np.square(filtered_rnoise))/float(len(filtered_lnoise)+len(filtered_rnoise))
# 	find total power in left and right channels
	pwr = lfpwr+rfpwr
	
#	background measures the background level of noise. It finds the minimum total power out of all 
# 	frames so far
	if background > pwr:
		background = pwr

	pwr_back = pwr/float(background)
#	if the ratio of power in the current frame to the background power level is greater than a threshold,
#	manually decided, consider it as a sound event
	if pwr_back>threshold:
		print 'listening...'
#		find the angle of the source using the time delay measured by cross correlation			
		angle1 = find_angle(filtered_lnoise,filtered_rnoise)


		if abs(angle1)>1:
	#		if the angle is sufficiently large, rotate counter-clockwise by  5 degrees and listen again
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
#			once again apply the filter on the left and right channels	
			filtered_lnoise = lfilter(b,a,leftch)
			filtered_rnoise = lfilter(b,a,rightch)
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
	
			print 'done rotating...'
		else: 
#		angle wasn't large enough - check up to the corresponding 'if' statement	
			print 'angle wasn''t large enough to rotate:',angle1
	else:	
#	sound wasn't loud enough - do nothing-check up to the corresponding 'if' statement	
		stream.stop_stream()
		stream.close()
		print 'no sound detected'
	print ' '
	



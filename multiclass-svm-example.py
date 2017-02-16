from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm
from sklearn.metrics import accuracy_score
from spectralFeatures_functions import make_single_channel,M,M_inv,H,mfcc_coeff, crossingRate,diff,sign,get_wave,spectral_centroid,spectral_flux,spectral_flatness,pwr,delta_mfcc_coeffs,runningMeanFast, stft, spec_plot, filterit,fourhertzmod, entropy1,quantize
from sklearn import cross_validation

fs = 44100.0
factor = 0.4
N  =500
L = 2**np.ceil(np.log2(N))
M = 250


# 0 for silence
# 1 for music
# 2 for speech


#print 'recording done!'
blockSize = L
frameTime = fs/blockSize

input_wave1 = get_wave('161825__xserra__carnatic-violin.wav',43)
input_wave2 = get_wave('203908__s9ames__bologna-speech-italian2.wav',43)
input_wave = np.concatenate((input_wave1,input_wave2),axis =0)

#input_wave1  = filterit(input_wave,500,'low',fs)
#input_wave2  = filterit(input_wave,500,'low',fs)

# entropy
entr1 = entropy1(input_wave1,factor,blockSize)
entr2 = entropy1(input_wave2,factor,blockSize)
entr = np.concatenate((entr1,entr2),axis =0)
entr = np.reshape(entr,(len(entr),1))

# zero crossing rate
crs1 = crossingRate(input_wave1,factor,blockSize)
crs2 = crossingRate(input_wave2,factor,blockSize)



crs = np.concatenate((crs1,crs2),axis=0)
crs = np.reshape(crs,(len(crs),1))
pwr1 = pwr(input_wave1,factor,blockSize)
pwr2 = pwr(input_wave2,factor,blockSize)
print 'crs'

# 4 Hz modulation energy
fourhz1 = fourhertzmod(input_wave1,factor,L)
fourhz2 = fourhertzmod(input_wave2,factor,L)
fourhz = np.concatenate((fourhz1,fourhz2),axis=0)
fourhz = np.reshape(fourhz,(len(fourhz),1))

print 'shape of four hertz vector',fourhz.shape
print 'shape of zero crossing vector',crs.shape

print 'mean music zcr',np.mean(crs1)
print 'mean speech zcr',np.mean(crs2)

# creating training data
frame_marker = np.concatenate(( np.array( [0 if pwr1[i]< 0.035 else 1 for i in range(len(crs1))] ), np.array( [0 if pwr2[i]< 0.035 else 2 for i in range(len(crs2))] ) ),axis = 0)

# mfcc coefficients
mfcc_coeffs1 = mfcc_coeff(input_wave1,factor,blockSize)
mfcc_coeffs2 = mfcc_coeff(input_wave2,factor,blockSize)
mfcc_coeffs = np.concatenate((mfcc_coeffs1,mfcc_coeffs2),axis = 0)

print 'mean music mfcc',np.ndarray.mean(mfcc_coeffs1,axis=0)
print 'mean speech mfcc',np.ndarray.mean(mfcc_coeffs2,axis=0)

# detla mfcc coefficients
delta_coeffs = delta_mfcc_coeffs(mfcc_coeffs)
delta_delta_coeffs = delta_mfcc_coeffs(delta_coeffs)


# spectral flux
spec_flux1 = spectral_flux(input_wave1,factor,blockSize)
spec_flux2 = spectral_flux(input_wave2,factor,blockSize)
spec_flux = np.concatenate((spec_flux1,spec_flux2),axis = 0)
spec_flux = np.reshape(spec_flux,(len(spec_flux),1))
print 'mean music spectral flux',np.mean(spec_flux1)
print 'mean speech spectral flux',np.mean(spec_flux2)

#print 'spec_flux'
#for i in spec_flux:		
#	print i



#print 'spec_flat'
#for i in spec_flat:#
#	print i
# final features used : spectral flux, zero crossing rate, mfcc coefficients, delat coefficients, delta delta coefficients.


print 'lengths of mfcc coeff vector',len(mfcc_coeffs),len(mfcc_coeffs[0])
print 'number of ones in the annotations (number of frames marked as speech)',sum(frame_marker), 'total number of frames',len(frame_marker)
clf = OneVsRestClassifier(LinearSVC(random_state=0))

# plot the spectrogram of data
X1,S = stft(input_wave,L,N,M)
t_spec = np.linspace(0,len(input_wave)/float(fs),X1.shape[0])
f_spec = np.linspace(0,fs,X1.shape[1])
#spec_plot(f_spec,t_spec,X1,'Frequency(Hz)','Time(s)','spectrogram')

# attempt pre-emphasis of the data
x_filt  = filterit(input_wave,500,'high',fs)

X_filt,S_filt = stft(x_filt,L,N,M)
t_spec = np.linspace(0,len(x_filt)/float(fs),X_filt.shape[0])
f_spec = np.linspace(0,fs,X_filt.shape[1])
#spec_plot(f_spec,t_spec,X_filt,'Frequency(Hz)','Time(s)','spectrogram')

#X = np.concatenate((spec_flux,crs,mfcc_coeffs,delta_coeffs,delta_delta_coeffs),axis=1)
X = np.concatenate((spec_flux,entr,crs,mfcc_coeffs,delta_coeffs,delta_delta_coeffs),axis=1)
X_scaled = preprocessing.scale(X)
y = frame_marker

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
					X_scaled, y,test_size = 0.4,random_state = 5
					)
clf.fit(X_train,y_train)

# test data 

testwav1 = get_wave('161825__xserra__carnatic-violin.wav',10)
#testwav1 = get_wave('164095__milton__crappy-violin.wav',10)
testwav2 = get_wave('189479__wordswar__soldier-s-speech.wav',10)
testwav = np.concatenate((testwav1,testwav2),axis =0)

pwr1_samp = pwr(testwav1,factor,blockSize)
pwr2_samp = pwr(testwav2,factor,blockSize)

# entropy
entr1_samp = entropy1(testwav1,factor,blockSize)
entr2_samp = entropy1(testwav2,factor,blockSize)
entr_samp = np.concatenate((entr1_samp,entr2_samp),axis =0)
entr_samp = np.reshape(entr_samp,(len(entr_samp),1))

# zero crossing rate
crs_samp1 = crossingRate(testwav1,factor,blockSize)
crs_samp2 = crossingRate(testwav2,factor,blockSize)
crs_samp = np.concatenate((crs_samp1,crs_samp2),axis = 0)
crs_samp = np.reshape(crs_samp,(len(crs_samp),1))

#mfcc coefficients
mfcc_samp1 = mfcc_coeff(testwav1,factor,blockSize)
mfcc_samp2 = mfcc_coeff(testwav2,factor,blockSize)
mfcc_samp = np.concatenate((mfcc_samp1,mfcc_samp2),axis = 0)

# four Hz modulation energy
fourhz_samp1 = fourhertzmod(testwav1,factor,blockSize)
fourhz_samp2 = fourhertzmod(testwav2,factor,blockSize)
fourhz_samp = np.concatenate((fourhz_samp1,fourhz_samp2),axis = 0)
fourhz_samp = np.reshape(fourhz_samp,(len(fourhz_samp),1))

# spectral flux
spec_flux1 = spectral_flux(testwav1,factor,blockSize)
spec_flux2 = spectral_flux(testwav2,factor,blockSize)
spec_flux_samp = np.concatenate((spec_flux1,spec_flux2),axis = 0)
spec_flux_samp = np.reshape(spec_flux_samp,(len(spec_flux_samp),1))

# delta coefficients
delta_coeffs_samp = delta_mfcc_coeffs(mfcc_samp)
delta_delta_coeffs_samp = delta_mfcc_coeffs(delta_coeffs_samp)

print len(crs_samp),len(mfcc_samp)

# test frame marker data to check the prediction accuracy
frame_marker_samp = np.concatenate((np.array([0 if pwr1_samp[i]< 0.035 else 1 for i in range(len(crs_samp1))]),np.array([0 if pwr2_samp[i]< 0.035 else 2 for i in range(len(crs_samp2))])),axis = 0)

print 'length of manual annotation vector: frame_marker_samp',len(frame_marker_samp)
cnt_0 = 0
cnt_1 = 0


#X_samp = np.concatenate((spec_flux_samp,crs_samp,mfcc_samp,delta_coeffs_samp,delta_delta_coeffs_samp),axis=1)
X_samp = np.concatenate((spec_flux_samp,entr_samp,crs_samp,mfcc_samp,delta_coeffs_samp,delta_delta_coeffs_samp),axis=1)
X_samp = preprocessing.scale(X_samp)
print 'dimensions of X_samp', len(X_samp),len(X_samp[0])

prediction1=clf.predict(X_test)
print prediction1
print sum(prediction1)
accuracy1 = accuracy_score(prediction1,y_test,normalize=True)
print 'accuracy (A)',accuracy1


prediction2 = clf.predict(X_samp)
print sum(runningMeanFast(prediction2,100))
#print 'range of quantized values',range(quantize(runningMeanFast(prediction2,100)))

accuracy2 = accuracy_score(quantize(runningMeanFast(prediction2,100)),frame_marker_samp,normalize=True)
print 'accuracy (B)',accuracy2

#---------------------------------------------------------------------#

t = np.linspace(0,len(testwav)/float(fs),len(frame_marker_samp))
t2 = np.linspace(0,len(testwav)/float(fs),len(testwav))
axes = plt.gca()
axes.set_xlim([0,20])
axes.set_ylim([-2,3])
#plt.plot(t,quantize(runningMeanFast(prediction2,100)),'g')
plt.plot(t,frame_marker_samp,'b')
plt.plot(t2,testwav,'r')
plt.show()



























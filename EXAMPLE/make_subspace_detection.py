from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime, read
from obspy.core.inventory.inventory import read_inventory
import os
import numpy as np
from numpy import linalg
import scipy.signal as signal   # For signal processing
import scipy.ndimage as ndimage
from scipy.stats import f
import scipy.stats
import h5py
import sys
import matplotlib.pyplot as plt   

## Make subspace detection for one day

# Pre-processes the data of this day
def pre_process(st):

    # Merge to make a one day record and interpolate gap
    st.merge(1,fill_value='interpolate')

    # Remove mean of the one day timeseries
    tr = st[0]
    tr.detrend(type='demean')
    if(tr.stats.sampling_rate == 100):
        pass
    else:
        if(tr.stats.sampling_rate > 100):
            tr.filter("lowpass", freq=40.0, corners=8, zerophase=True)

            if(np.mod(tr.stats.sampling_rate,100) == 0):
                tr.decimate(int(round(tr.stats.sampling_rate/100.)),no_filter=True)
            else:
                tr.interpolate(100,"weighted_average_slopes")

        else:
            tr.interpolate(100,"weighted_average_slopes")

    return tr

# Set the year and the day number
year = int(sys.argv[1])
julday = int(sys.argv[2])
print('Running detections on %d/%03d'%(year,julday))
sys.stdout.flush()
sys.stderr.flush()

# Get the time of start and finish
stime = UTCDateTime(year=year,julday=julday)
etime = stime  + 86400
stime0=  stime - 90
etime0 = stime  + 86400 + 90

# Select the client
clientPF=Client('IPGP')

# Set the size of the window
ns = 1024

# Set the number of singular vectors that we use for detection
d = [5,4]

# Stations 
sta=['PROZ','PRON','PROE','CILZ', 'CILN', 'CILE','MVLZ','MVLE', 'MVLN', 'MAIDZ', 'MAIDE', 'MAIDN']
nsta = 4 # number of stations used for detection

# Max memory per instance
maxmem  = 50. # In GB
#maxnpts = int(maxmem*1e9/(2*4*ns)+ns) # We need to store 2 matrices of size (npts-ns) x ns
 
# Set sampling frequency
dt = 0.01
fs = int(1./dt)

# Set the number of samples and matrix we use for detection
nhours = 24
npts = nhours*3600*fs
maxnpts = 720001
nb_matrix = int((npts-ns)/maxnpts)

# Build the frequency vector and the filter
fmin = 8.
fmax = 32.
freq_c = [fmin, fmax]
freq_c = np.array(freq_c)
Wn = freq_c * 2. * dt # Normalizing frequencies
sos = signal.butter(4, Wn, 'bandpass', output='sos')

# Build the tapering function in the x dimension
taper_width = 0.1*float(ns)/float(npts)
ts = signal.tukey(npts,taper_width)

# Create empty lists to store data for all clusters
all_cdec = np.zeros(npts,dtype='float32') # values of statistic exceeding the threshold
all_tdec = np.zeros(npts,dtype='float32') # times of detection
cl = np.zeros(npts,dtype='int32') # cluster ids
threshold = np.zeros(npts,dtype='float32') # values of threshold 

# Tolerance on picking times
tt_tol = 5 # 5 x 1e-2 sec

# Set standard deviation coefficient for threshold determination
D = 23

# Create time vector
t = np.arange(npts,dtype='int32')

# Loop over clusters
iicl = 0
for icl in [18,25]: 
    
    # Create a list to store statistic values for all stations 
    all_mc = np.zeros(npts-ns,dtype='float32')
    
    # Loop over stations
    ista = 0
    l=0
    while ista !=nsta:
    
        # Importation of the continuous data stream at one station
        t1=UTCDateTime.now()
        test = False
        while test == False:
            try:
                st = clientPF.get_waveforms('PF',sta[l][0:-1],'*','HH'+sta[l][-1], stime0, etime0)
                test = True
            except Exception as e:
                assert l<len(sta), 'Cannot run detection on year %d day %d - Not enough data'%(year,julday)
                print('Cannot download %s on year %d day %d'%(sta[l],year,julday))
                l = l+1
                continue
        print('Data download elapsed time : %.2f sec' %(UTCDateTime.now()-t1))
       
        # Pre-process waveform
        st = pre_process(st)
	
        # Trim waveform
        st.trim(stime,etime,nearest_sample=False,pad=True)
        if st.stats.endtime.day==st.stats.starttime.day+1:
            st.trim(stime,etime-0.01,nearest_sample=False) # We assume 100 sps data
        print('Pre-processing/triming elapsed time : %.2f sec' %(UTCDateTime.now()-t1))

        stream = st.data
        assert st.stats.delta == dt, 'Incorrect sampling step'
        assert stream.size == npts, 'Wrong number of samples on %s year %d day %d' %(sta[l],year,julday)
        sys.stdout.flush()
        sys.stderr.flush()
       
        # Filter the data stream
        stream = stream - np.mean(stream)
        stream = signal.sosfilt(sos,ts*(stream))*ts
        
        # Extract singular vectors and P arrival time for time-shift
        T = open('list_cluster','rt').readlines()
        h = h5py.File(T[icl][:-1],"r")
        print(h)
        path = '/STATIONS/'+sta[l]

        # Test if the station exists in the file
        e = path in h
        if(e):
            TW = h[path].attrs['TW']
            U = np.zeros((ns,d[iicl]),dtype='float32')
            for j in range(d[iicl]):
                attrsname = 'w_{}'.format(j)
                U[:,j] = h[path].attrs[attrsname] # subspace representation matrix
            
            # Scan data stream and get statistic values
            c = np.zeros(npts-ns,dtype='float32')
            UUt = U.dot(np.transpose(U))
            
            # Calculate c
            t1=UTCDateTime.now()
            flagcut = False
            if npts>maxnpts:
                npts
                flagcut = True
            if flagcut:
                print('Subspace scan done in %s matrix products' %(nb_matrix+1))
                kk = 0
                for j in range(nb_matrix):
                    stream_w = np.zeros((maxnpts,ns),dtype='float32')
                    for k in range(maxnpts):
                        stream_w[k,:]=stream[kk+k:kk+k+ns]
                    kk = kk+k+1
                    newc = np.sum((UUt.dot(stream_w.T))**2,axis=0)/np.sum(stream_w**2,axis=1)
                    c[j*maxnpts:j*maxnpts+maxnpts] = newc
                
                stream_w = np.zeros((npts-ns-maxnpts*nb_matrix,ns),dtype='float32')
                for k in range(npts-ns-maxnpts*nb_matrix):
                    stream_w[k,:]=stream[kk+k:kk+k+ns]
                newc = np.sum((UUt.dot(stream_w.T))**2,axis=0)/np.sum(stream_w**2,axis=1)
                c[kk:] = newc

            else:
                print('Subspace scan done in 1 matrix products')
                stream_w = np.zeros((npts-ns,ns),dtype='float32')
                for k in range(npts-ns):
                    stream_w[k,:]=stream[k:k+ns]
                c = np.sum((UUt.dot(stream_w.T))**2,axis=0)/np.sum(stream_w**2,axis=1)
            print('Subspace scan elapsed time : %.2f sec' %(UTCDateTime.now()-t1))
            
            # Maximum filter
            c = ndimage.maximum_filter1d(c,tt_tol)
           
            # Replace non valid values with 0
            nan = np.isnan(c)
            inan = np.where(nan==True)[0]
            for n in inan:
                c[n]=0 

            # Time-shift + stacking
            t1=UTCDateTime.now()
            assert int(TW/dt)>0, 'Negativ TW at %s' %(sta)
            all_mc = all_mc + np.roll(c,-int(TW/dt))
            print('Time shift + stacking elapsed time : %.2f sec' %(UTCDateTime.now()-t1))
        
        h.close()
        ista = ista+1
        l = l+1
    iicl = iicl+1
    
    # Take the mean
    all_mc = all_mc/np.float64(nsta)
      
    # Threshold determination
    gamma = np.std(all_mc)*D

    # Scan statistic values to look for events
    t1=UTCDateTime.now()
    idec = np.where(all_mc>gamma)[0]
    for i in idec:
          if t[i] not in all_tdec:
            all_tdec[i] = t[i]
            all_cdec[i] = all_mc[i]
            cl[i] = icl
            threshold[i] = gamma

          else:
            if all_mc[i]>all_cdec[i]:
                all_cdec[i] = all_mc[i]
                cl[i] = icl
                threshold[i] = gamma
    print('Statistic values scan elapsed time : %.2f sec' %(UTCDateTime.now()-t1))

# Only keep non-null values
all_cdec = [x for x in all_cdec if x!=0]
all_tdec = [x for x in all_tdec if x!=0]
cl = [int(x) for x in cl if x!=0]
threshold = [x for x in threshold if x!=0]

# Write data in a file
f = open("D_%4d_%03d_%02.0f_%02.0f_%.0fM" %(year,julday,fmin,fmax,D),"w")
for i in range(len(all_cdec)):
    f.write('%4d %10d %17.12f %17.12f\n'%(cl[i]+1,all_tdec[i],all_cdec[i],threshold[i]))
f.close()

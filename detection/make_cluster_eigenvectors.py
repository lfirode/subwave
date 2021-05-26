import numpy as np
import h5py
import scipy.signal as signal   # For signal processing

data_saved = np.load('clustering_0.75_2.npz')
clusters = data_saved['name2']
cl =  data_saved['name1']

T=open('list_det','rt').readlines()
# list all templates from the template list used in = H5F2DT

# Stations
stas_w = ['PROZ', 'PROE', 'PRON', 'CILZ', 'CILE', 'CILN', 'MVLZ', 'MVLE', 'MVLN', 'MAIDZ', 'MAIDE', 'MAIDN']
stas_nw = ['CAMZ', 'CAMN', 'CAME', 'BLEZ', 'PJRZ', 'RIV5Z', 'RIV5N', 'RIV5E', 'RMA3Z', 'RMA3E', 'RMA3N', 'RIV4Z', 'RIV4E', 'RIV4N', 'RMA1Z', 'RMA1E', 'RMA1N']

# Set the size of the seismograms
ns = 1024

# Set sampling frequency
dt = 0.01
fs = int(1./dt)
# Build the tapering function in the x dimension (size of the window)
ts = signal.tukey(ns,0.1)
# Build the frequency vector and the filter
fmin = 8.
fmax = 32.
freq_c = [fmin, fmax]
freq_c = np.array(freq_c)
Wn = freq_c * 2. * dt # Normalizing frequencies
sos = signal.butter(4, Wn, 'bandpass', output='sos')

# Loop over the clusters id
for iicl, icl in enumerate(cl):
    I=np.where(clusters==icl)[0] 

    # Build and empty hdf5 file
    lat= []
    lon=[]
    depth=[]
    for ievt in I:
        h1 = h5py.File(T[ievt][:-1], "r")
        lat.append(h1['/HEADER'].attrs['LAT'])
        lon.append(h1['/HEADER'].attrs['LONG'])
        depth.append(h1['/HEADER'].attrs['DEPTH'])
        h1.close()

    # Make a first group (HEADER)
    hdf5filename =  'cluster_{0:04d}.hdf5'.format(iicl) 
    print(hdf5filename)
    # Open the hdf5 file that we want to create
    h = h5py.File(hdf5filename, "w")

    grp_header = h.create_group("HEADER")
    # We put in this group all the following informations 
    grp_header.attrs['ID'] = np.string_(str(iicl))
    grp_header.attrs['YEAR'] = 1999
    grp_header.attrs['MONTH'] = 1
    grp_header.attrs['DAY'] = 1
    grp_header.attrs['HOUR'] = 1
    grp_header.attrs['MIN'] = 1
    grp_header.attrs['SEC'] = 1
    grp_header.attrs['LAT'] = np.mean(lat)
    grp_header.attrs['LONG'] = np.mean(lon)
    grp_header.attrs['DEPTH'] = np.mean(depth)
    grp_header.attrs['MAG'] = 0.0
    grp_data = h.create_group("STATIONS")
    
    # Loop over the stations
    for sta in stas_w:
        W = np.zeros((len(I),ns))
        W0 = np.zeros((len(I),ns))
        AA = np.zeros(len(I))
        TP = np.zeros(len(I))
        for ievt, evt in enumerate(I):
            h1 = h5py.File(T[evt][:-1], "r")
            path0 = '/STATIONS/' + sta
            # Test if the station exists in the second file
            e = path0 in h1
            if(e):
                # Set the path to the current station trace
                path = path0 + '/Trace'
                # Extract the waveforms
                y1 = np.array(h1[path])
                y1 = y1 - np.mean(y1)
                y1_f = signal.sosfilt(sos,ts*(y1))*ts
                W[ievt,:] = y1_f
                W0[ievt,:]  = y1
                AA[ievt] = np.std(np.abs(y1_f[110:]))/np.std(np.abs(y1_f[:90])) # SNR
                TP[ievt] = h1[path0].attrs['TP']
                npre = h1[path].attrs['Np'] 
            h1.close()

        # Get only events that have a pick at this staion    
        J = np.all(W==0,axis=1)
        W = W[~J,:] ;  AA = AA[~J]; TP = TP[~J];  W0 = W0[~J,:]
        # Get only events that pass SNR threshold
        J = np.where(AA > 1.25)[0]
        W = W[J,:] ;  AA = AA[J]; TP = TP[J];  W0 = W0[J,:]
        # Number of valid waveforms
        nwav = len(AA)
        # If there is more than 1 record at this station
        if(nwav>=2):
            # First find the event that correlates the best with the other ones
            CR = np.zeros(nwav)
            for iw1 in range(nwav):
                for iw2 in range(nwav):
                    R = signal.correlate(W[iw1,:], W[iw2,:], mode='full')  / (ns* np.std(W[iw1,:]) * np.std(W[iw2,:]))
                    CR[iw1] = CR[iw1] + np.amax(R)
            # This event will serve as a reference event
            CRmax = np.argmax(CR)
            Ws = np.zeros((nwav,ns))
            # Compute time shift relative to this event
            for iw in range(nwav):
                R = signal.correlate(W[CRmax,:], W[iw,:], mode='full')   / (ns* np.std(W[CRmax,:]) * np.std(W[iw,:]))
                delta_t = np.argmax(R)-(ns-1)
                #if((np.amax(R) > 0.6) and (np.abs(delta_t) < 15)) :
                Ws[iw,:] = np.roll(W[iw,:],delta_t) 
            # Create a group for each station
            grp_sta = grp_data.create_group(sta)
            # Add TP and TW as current attributes to the station
            grp_sta.attrs['TP'] = np.mean(TP)
            grp_sta.attrs['TW'] = np.mean(TP) - npre*dt
            # Singular vectors
            X = np.transpose(Ws)
            w, s, v = np.linalg.svd(X)         
            # Store the singulars vectors and values in a hdf5file
            for k in range(len(w)):
               attrsname = 'w_{}'.format(k)
               grp_sta.attrs[attrsname] = w[:,k]
            grp_sta.attrs['s'] = s
    
    for sta in stas_nw:
        TP = np.zeros(len(I))
        for ievt, evt in enumerate(I):
            h1 = h5py.File(T[evt][:-1], "r")
            path0 = '/STATIONS/' + sta
            # Test if the station exists in the second file
            e = path0 in h1
            if(e):
                TP[ievt] = h1[path0].attrs['TP']
            h1.close()
        # Only keep valid picks
        TP = TP[TP>0.]
        if(len(TP)>0):
            # Create a group for each station
            grp_sta = grp_data.create_group(sta)
            grp_sta.attrs['CMPNT'] = sta[-1:]
            # Add TP and TW as current attributes to the station
            grp_sta.attrs['TP'] = np.mean(TP)
            grp_sta.attrs['TW'] = np.mean(TP) - npre*dt
            grp_sta.attrs['W'] = 0




    h.close()

    

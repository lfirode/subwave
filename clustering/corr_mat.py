#!/usr/bin/env python
# Reads dt.cc and removes pairs with too few stations
import sys
import numpy as np

# Transform the dt file into the mean cc for each doublet

# Input parameters
minsta = 3      # Min number of stations

ifile = sys.argv[1] # Input dt.cc file
ofile = sys.argv[2] # Output file with average cc for each doublet

print(ifile)

# Read input dt.cc file
L = open(ifile,'rt').readlines()

# Clean up and write output file
f = open(ofile,'wt')
lines = []
for i,l in enumerate(L):
    if l[0]=='#':
        if(i>0):
            if(len(MCC)>=minsta):
                RC = np.mean(MCC)
                out = i1 + " " + i2 + " " + str(RC) + '\n'
                f.write(out)
        MCC = []
        items = l.strip().split()
        i1 = items[1]
        i2 = items[2]
    else:
        items = l.strip().split()
        cc = float(items[2])
        MCC.append(cc)
        
f.close()

import matplotlib.pyplot as plt
import numpy as np
import os
import csv 

if os.path.exists("maxerr.txt"):
  os.remove("maxerr.txt")
else:
  print("The file does not exist")


dls = np.logspace(-3, 0, num=4)

for dl in dls:
    os.system('python3 snellvpol.py '+str(dl)+' 0 0')

with open('maxerr.txt') as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    # next(reader, None)  # skip the headers
    data_read = np.array([[float(y) for y in x] for x in reader])

data_read = data_read.T

plt.loglog(data_read[0,:], data_read[1,:], '.-')
plt.show()

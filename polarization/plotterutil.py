import numpy as np
import matplotlib.pyplot as plt
import snell_fns as sf

z=np.linspace(10, -500, num=1000)
ang = np.random.random()*np.pi*2
plt.plot(z, [sf.npp(ang, z[i]) for i in range(len(z))])
plt.xlim([z[0], z[-1]])
plt.show()


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

NUM_SAMPLES = 10000
parameters = np.zeros((NUM_SAMPLES,4))

for k in range(NUM_SAMPLES):
  if k % 100 ==0:
    print(k)

  fig, ax = plt.subplots(figsize=(8,8),frameon=False)

  m = np.vstack((np.random.uniform(-1,1,2), np.random.uniform(-1,1,2)))
  cov = np.dot(m,np.transpose(m))
  x, y = np.random.multivariate_normal([0, 0], cov, 50000).T

  ax.hist2d(x,y,bins=64,cmap='Greys_r');
  #ax.axis('equal');
  ax.axis('off')

  plt.subplots_adjust(wspace=0, hspace=0, left=0,bottom=0, top=1, right=1)

  parameters[k,:] = np.round(cov.flatten(),4).tolist()
  plt.savefig('/scratch/rag394/data/gaussian_generator/gaussian_{}.png'.format(k))
  plt.close();

pd.DataFrame(parameters,index=range(NUM_SAMPLES)).to_csv('/scratch/rag394/data/gaussian_generator/gaussian_parameters.csv')

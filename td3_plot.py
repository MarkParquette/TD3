##
## td3_plot.py
##
## Utility that plots the results from the TD3 main program.
##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_results(name, policy_name="TD3", eval_freq=5000):
  data = np.load(f"./results/{name}.npy")

  x = []  
  for i in range(len(data)):
    x.append(i*eval_freq)

  #sns.set()
  fig = plt.figure(figsize = (12, 7))
  #fig.set_size_inches(2.0,4.0)
  plt.title(f"{name}")
  plt.xlabel("Timesteps")
  plt.ylabel("Ave. Reward")
  plt.plot(x, data, label=f"{policy_name}")

  plt.legend()
  plt.show()

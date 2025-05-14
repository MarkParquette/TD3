##
## td3_plot.py
##
## Utility that plots the results from the TD3 main program.
##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sliding_average(data, window_size):
    """
    Calculates the sliding average of a NumPy array.

    Args:
        data (np.ndarray): The input array.
        window_size (int): The number of elements to include in the moving window.

    Returns:
        np.ndarray: An array containing the sliding averages.
                      The length of the output array will be less than the input array by window_size - 1.
    """
    if window_size > len(data):
      window_size = len(data)

    # Create an array of ones to act as weights for the average
    weights = np.ones(window_size) / window_size

    # Use convolution to calculate the sliding average
    # mode='valid' returns only the parts where the window fully overlaps with the input
    return np.convolve(data, weights, mode='valid')

def plot_results(base_name, policy_name="TD3", eval_freq=5000, window_size=10, show_train=True, show_ave=True):
  plt.figure(figsize = (12, 7))
  plt.title(f"{base_name}")
  plt.xlabel("Timesteps")
  plt.ylabel("Ave. Reward")

  policy_names = ["TD3", "TD3-DEV"]#, policy_name]

  for policy in policy_names:
    try:
      data = np.load(f"./results/{policy}_{base_name}.npy")
    except:
      continue
    
    ave = sliding_average(data, window_size)
    offset = min(len(data), window_size) * eval_freq

    x = []  
    for i in range(len(data)): x.append(i*eval_freq)
    if show_train: plt.plot(x, data, label=f"{policy}")

    x_ave = []  
    for i in range(len(ave)): x_ave.append(i*eval_freq + offset)
    if show_ave: plt.plot(x_ave, ave, label=f"{policy} (Ave)")

    max = np.max(data)
    max_data = []
    for i in range(len(data)): max_data.append(max)
    plt.plot(x, max_data, label=f"{policy} (Max) {max:.2f}")

  solved = 200.
  solved_data = []
  for i in range(len(data)): solved_data.append(solved)
  #plt.plot(x, solved_data, label=f"Solved Level {solved:.2f}")

  plt.legend()
  plt.show()

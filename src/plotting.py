import keras
import matplotlib.pyplot as plt
import numpy as np

def plot_impedance(n,
                   valIp,
                   X_predict):
    plt.figure(figsize=(10, 15))
    trace_size = len(valIp[0, :, 0])
    ip_end = valIp[0,:,0].max()
    for i in range(1, n + 1):
        ax = plt.subplot(2, n, i)
        plt.tight_layout()
        ax.set_title("Trace {0}".format(i-1))
        plt.plot(valIp[i-1, :, 0], range(trace_size))
        plt.plot(X_predict[:, i-1], range(trace_size), '--r')
        plt.legend(['original Ip', 'predicted Ip'])
        plt.ylabel("Time (ms)")
        plt.xlabel("Impedance ((km/s).(gm/cm³))")
        ax.set_ylim(0, trace_size)
        ax.set_xticks(np.arange(0, ip_end+0.1, 0.2))
        ax.set_yticks(np.arange(0, trace_size+1, 20))
        ax.invert_yaxis()
    plt.show()
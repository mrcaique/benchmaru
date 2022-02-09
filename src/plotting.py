import keras
import matplotlib.pyplot as plt

def plot_impedance(n,
                   valIp,
                   X_predict):
    plt.figure(figsize=(20, 10))
    for i in range(1, n + 1):
        ax = plt.subplot(2, n, i)
        ax.set_title("y_validation {0}".format(i-1))
        plt.plot(valIp[i-1, :, 0])
        plt.plot(X_predict[:, i-1], '--r')
        plt.legend(['original Ip', 'predicted Ip'])
        plt.xlabel("Time (s)")
        plt.ylabel("Impedance ((km/s).(gm/cm³))")
    plt.show()
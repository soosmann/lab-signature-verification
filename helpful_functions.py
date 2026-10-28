import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import det_curve


def calculate_eer(far, frr):
    min_index = np.nanargmin(np.abs(far - frr))
    eer = (far[min_index] + frr[min_index]) / 2
    return eer

def print_errors(Ns, GenuineScores_list, ImpostorScores_list):
    for idx, N in enumerate(Ns):
        genuine = GenuineScores_list[idx].flatten()
        impostor = ImpostorScores_list[idx].flatten()
        
        scores = np.concatenate([genuine, impostor])
        labels = np.concatenate([np.ones(len(genuine)),
                                 np.zeros(len(impostor))])
        
        far, frr, thresholds = det_curve(labels, scores)
        eer = calculate_eer(far, frr)
        print(f"EER for N={N}: {eer*100:.2f}%")

def plot_det_curve(Ns, GenuineScores_list, ImpostorScores_list):
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green']
    
    for idx, N in enumerate(Ns):
        genuine = GenuineScores_list[idx].flatten()
        impostor = ImpostorScores_list[idx].flatten()
        
        scores = np.concatenate([genuine, impostor])
        labels = np.concatenate([np.ones(len(genuine)),
                                 np.zeros(len(impostor))])
        
        far, frr, _ = det_curve(labels, scores)
        eer = calculate_eer(far, frr)
        
        plt.plot(far * 100, frr * 100,
                 color=colors[idx],
                 label=f'N={N} | EER={eer*100:.2f}%')
    
    plt.xlabel("False Acceptance Rate (%)")
    plt.ylabel("False Rejection Rate (%)")
    plt.legend(loc="upper right")
    
    plt.yscale('log')
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter(10, labelOnlyBase=False))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatter(10, labelOnlyBase=False))
    plt.grid(which='both')
    plt.axis([0.1,100,0.1,100])
    
    plt.show()
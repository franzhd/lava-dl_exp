import os
import sys
from typing import Any
import matplotlib.pyplot as plot
import numpy as np
import torch
import torch.nn.functional as F
import lava.lib.dl.slayer as slayer
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class SpikeRatePlus(torch.nn.Module):
    
    def __init__(self,true_rate,false_rate, max_rate=0.3,device='cpu') -> None:
        
        super(SpikeRatePlus, self).__init__()
        self.spike_rate_loss = slayer.loss.SpikeRate(true_rate=true_rate, false_rate=false_rate, reduction='sum').to(device)
        if  max_rate > 1:
            raise AssertionError(
                f'Expected max rate to be between 0 and 1. Found {max_rate}'
            )
        self.max_rate = max_rate


    def forward(self, output_list, labels):
        spike, event_rate = output_list
        loss_1 = self.spike_rate_loss.forward(spike, labels)
        loss_2 =abs(event_rate - self.max_rate)**2/2

        return loss_1 + loss_2


        
        

def compute_output_labels(matrix):
    # Sum the elements along the last dimension
    #print(f'compute_output_labels matrix shape {matrix.shape}')
    summed_matrix = np.sum(matrix, axis=-1)
     
    # Divide the sum by the number of elements in the second dimension
    divided_matrix = summed_matrix / matrix.shape[1]

    # Find the maximum value along the second dimension and its index
    max_indices = np.argmax(divided_matrix, axis=1)

    return max_indices


def gen_confusion_matrix(predictions, labels, path):
    
    num_label = max(labels)
    #print(f'prediction shape {predictions.shape} and labels shape{len(labels)}')
    conf_matrix = confusion_matrix(predictions, labels)
    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(num_label),
                yticklabels=range(num_label))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Save the plot
    file_path = os.path.join(path,'confusion_matrix.png')
    plt.savefig(file_path, bbox_inches='tight')





def spike_plot(data,spike_data,  label, save=None):
    

    line_size = 0.5
    colors_spike = []
    color = []
    handles = []
    
    if len(spike_data.shape) == 3:
        for i in range(spike_data.shape[0]):
            r = np.random.random()
            b = np.random.random()
            g = np.random.random()
            color.append([r, g, b])
            for d in range(spike_data.shape[1]):
                colors_spike.append([r, g, b])
        
        plot_data =  spike_data.reshape((spike_data.shape[0]*spike_data.shape[1], spike_data.shape[2])) * np.arange(spike_data.shape[-1]) * 1.0/spike_data.shape[-1]

    if len(spike_data.shape) == 2:
        for i in range(spike_data.shape[0]):
            r = np.random.random()
            b = np.random.random()
            g = np.random.random()
            color.append([r, g, b])
            colors_spike.append([r, g, b])
            
        plot_data =  spike_data * np.arange(spike_data.shape[-1]) * 1.0/spike_data.shape[-1]
        print(f'plot data {plot_data.shape}')
    fig, (ax1, ax2) = plot.subplots(2, 1, figsize=(40, 30))
    fig.suptitle(f'Signlal label {label}')


    for i in range(data.shape[0]):
        ax1.plot(range(data.shape[-1]), data[i, :], color = color[i])
        handles.append(plot.Line2D([0], [0], color=color[i], lw=2, label=f'dimension {i}'))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Original signal dimensions')

    # print(f'colors spike {len(colors_spike)}')
    # print(f"plot data shape {plot_data.shape}")

    ax2.eventplot(plot_data, color=colors_spike, linelengths = line_size)     
    ax2.set_xlabel('Spike')
    ax2.set_ylabel('Channels')
    ax2.set_title('Spike Train encoding')
    plot.xlabel('Spike')
    plot.ylabel('Channels')

    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 0.85))
    if save is None:
        plot.show()
    else:
        plot.savefig(save)



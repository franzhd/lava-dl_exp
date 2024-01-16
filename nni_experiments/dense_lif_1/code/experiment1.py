import nni
import argparse
import os, sys
sys.path.insert(1, '../../../src/')
from pathlib import Path
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from torch.utils.data import  DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

from dataloader import WISDM_Dataset_parser, WISDM_Dataset
from utils import *
from encoding import *

def event_rate_loss(x, max_rate=0.01):
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))

class Network(torch.nn.Module):
    def __init__(self, input, output, treshold, voltage_decay, graded_spike):
        super(Network, self).__init__()

        neuron_params = {
                'threshold'     : treshold,
                'current_decay' : 1,               
                'voltage_decay' : voltage_decay,
                'tau_grad'      : 1,
                'scale_grad'    : 0.5,
                'graded_spike'  : False
            }
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.1),}
        neuron_params_drop['graded_spike'] = graded_spike
        
        self.blocks = torch.nn.ModuleList([

                slayer.block.cuba.Dense(neuron_params_drop, input, 128, weight_norm=True, delay=True),
                slayer.block.cuba.Dense(neuron_params_drop, 128,256,weight_norm=True,  delay=True),
                slayer.block.cuba.Dense(neuron_params_drop, 256,128, weight_norm=True, delay=True),
                slayer.block.cuba.Dense(neuron_params, 128, output)
            ])


    def forward(self, x):
        count = []
        event_cost = 0

        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = self.blocks[2](x)
        x = self.blocks[3](x)
        return x

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad
    

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', type=str, help='nome del config file per creare la cartella adeguata')
    args = parser.parse_args()

    params = nni.get_next_parameter()

    os.chdir(f'{Path.home()}/lava-dl_exp/nni_experiments/{args.trial_path}/{nni.get_experiment_id()}/trials/{nni.get_trial_id()}')
    trained_folder = 'Trained'
    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs('jpg', exist_ok=True)
    os.makedirs('gifs', exist_ok=True)
    

    dataset = WISDM_Dataset_parser(f'{Path.home()}/lava-dl_exp/data/watch_subset2_40.npz', norm=None)
    train_set = dataset.get_training_set()
    test_set = dataset.get_validation_set()

    data, label = train_set
    print(data.shape)

    #transform = transforms.Compose([oversample(3)])

    train_dataset = WISDM_Dataset(train_set)
    test_dataset = WISDM_Dataset(test_set)

    train_loader = DataLoader(dataset=train_dataset, batch_size=int(params['batch_size']), shuffle=True, num_workers=8)
    test_loader  = DataLoader(dataset= test_dataset, batch_size=int(params['batch_size']), shuffle=True, num_workers=8)

    for batch in train_loader:
        input, label = batch
        print(input.shape)
        break


    # x, y = train_set
    
    # for i in range(5):

    #     data, label  =  train_set
    #     r = np.random.randint(len(data))
    #     data, label = data[r], label[r]
    #     spike_plot(data, spike_data.reshape(data.shape[0], 2, spike_data.shape[1]), label, f'./jpg/lookup{i}_sod.jpg')

    device = torch.device('cuda')

    graded_spike = True if params['graded_spike'] == 1 else 0
    net = Network(6, 7, params['treshold'], params['voltage_decay'], graded_spike).to(device)

    optimizer = torch.optim.RAdam(net.parameters(), lr=params['lr'], weight_decay=1e-5)

    error = slayer.loss.SpikeRate(true_rate=0.80, false_rate=0.2, reduction='sum').to(device)
    
    stats = slayer.utils.LearningStats()
    
    assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)
    
    epochs = 50
    count = 0
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        labels = []
        outputs = []
        if count < 4:
            tqdm_dataloader = tqdm(train_loader)
            for _, batch in enumerate(tqdm_dataloader): # training loop
                input, label = batch
                output = assistant.train(input.to(device), label)
                tqdm_dataloader.set_description(f'\r[Epoch {epoch:2d}/{epochs}] {stats}')
            
            tqdm_dataloader = tqdm(test_loader)
            for _, batch in enumerate(tqdm_dataloader): #eval loop
                input, label = batch
                print(label.shape)
                output = assistant.test(input, label)
                tqdm_dataloader.set_description(f'\r[Epoch {epoch:2d}/{epochs}] {stats}')
                labels = labels + label.tolist()
                outputs = outputs + output.tolist()


            nni.report_intermediate_result(stats.testing.accuracy*100)
            count = count + 1 
            if stats.testing.best_accuracy:

                labels = np.array(labels).flatten()
                outputs = np.array(outputs)
                predictions = compute_output_labels(outputs)
                gen_confusion_matrix(predictions,labels, trained_folder)
                count = 0
                torch.save(net.state_dict(), trained_folder + '/network.pt')



            stats.update()
            stats.save(trained_folder + '/')
            net.grad_flow(trained_folder + '/')

 
    
    nni.report_final_result(stats.testing.max_accuracy*100)
    stats.plot(figsize=(15, 5),path=f'{trained_folder}/')
    
    
    net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    net.export_hdf5(trained_folder + '/network.net')

    output = net(input.to(device))
    for i in range(5):
        try:
            inp_event = slayer.io.tensor_to_event(input[i].cpu().data.numpy().reshape(1, 6, -1 ))
            out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 7, -1))
            inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
            out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
            inp_anim.save(f'./gifs/inp{i}_label{label[i]}.gif', animation.PillowWriter(fps=24), dpi=300)
            out_anim.save(f'./gifs/out{i}_label{label[i]}.gif', animation.PillowWriter(fps=24), dpi=300)
        except:
            pass

if __name__ == '__main__':
    main()
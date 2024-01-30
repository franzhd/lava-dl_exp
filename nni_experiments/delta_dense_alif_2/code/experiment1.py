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
from lava.lib.dl.slayer.utils.utils import event_rate
from dataloader import WISDM_Dataset_parser, WISDM_Dataset
from utils import *
from encoding import *


class Network(torch.nn.Module):
    def __init__(self, input, output, delta_treshold, delta_scale, treshold, voltage_decay, threshold_step, threshold_decay, refractory_decay, max_event_rate):
        super(Network, self).__init__()

        self.sparsity_loss = slayer.loss.SparsityEnforcer(max_rate=max_event_rate, lam=1.0)

        cuba_dense_neuron_params_input = {
                'threshold'     : treshold,
                'current_decay' : 1,               
                'voltage_decay' : voltage_decay,
                'tau_grad'      : 1,
                'scale_grad'    : 0.01,
                'requires_grad' : True,
                }
        
        cuba_dense_neuron_params = {
                'threshold'     : treshold,
                'current_decay' : 1,               
                'voltage_decay' : voltage_decay,
                'tau_grad'      : 1,
                'scale_grad'    : 1,
                'requires_grad' : True,
                }
        
        alif_dense_neuron_params = {
                'threshold'     : treshold,
                'current_decay' : 1,               
                'voltage_decay' : voltage_decay,
                'tau_grad'      : 1,
                'scale_grad'    : 1,
                'threshold_step': threshold_step,
                'threshold_decay': threshold_decay,
                'refractory_decay': refractory_decay
                # 'shared_param'   : False,
                # 'requires_grad' : True,
                # 'graded_spike'  : True

                }
        self.encoding = slayer.axon.delta.Delta(delta_treshold, scale=delta_scale)
        self.blocks = torch.nn.ModuleList([
                
                slayer.block.cuba.Dense(cuba_dense_neuron_params_input, input, 256, weight_norm=True),
                slayer.block.alif.Dense(alif_dense_neuron_params, 256,512,weight_norm=True),
                slayer.block.alif.Dense(alif_dense_neuron_params, 512,256, weight_norm=True),
                slayer.block.cuba.Dense(cuba_dense_neuron_params, 256, output)
            ])


    def forward(self, x):

        self.sparsity_loss.clear()
        x = self.encoding(x)
        for block in self.blocks:
            x = block(x)
            self.sparsity_loss.append(x)

        return x, self.sparsity_loss.loss

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
        print('input shape:', input.shape)
        break


    # x, y = train_set
    
    # for i in range(5):

    #     data, label  =  train_set
    #     r = np.random.randint(len(data))
    #     data, label = data[r], label[r]
    #     spike_plot(data, spike_data.reshape(data.shape[0], 2, spike_data.shape[1]), label, f'./jpg/lookup{i}_sod.jpg')

    device = torch.device('cuda')

    #graded_spike = True if params['graded_spike'] == 1 else 0
    net = Network(6, 7,params['delta_treshold'],params['delta_scale'], params['treshold'], params['voltage_decay'], params['treshold']*params['threshold_step'], params['threshold_decay'], params['refractory_decay'], params['max_rate']).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    error = slayer.loss.SpikeRate(true_rate=0.80, false_rate=0.2, reduction='sum').to(device)
    
    stats = slayer.utils.LearningStats()
    
    assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict, lam=1)

    epochs = 50
    count = 0
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        labels = []
        outputs = []
        if epoch % 15 == 0:
            assistant.reduce_lr()
        if count < 3:
            count = count+1
            tqdm_dataloader = tqdm(train_loader)
            for _, batch in enumerate(tqdm_dataloader): # training loop
                input, label = batch
                output = assistant.train(input.to(device), label)
                tqdm_dataloader.set_description(f'\r[Epoch {epoch:2d}/{epochs}] {stats}')
            
            tqdm_dataloader = tqdm(test_loader)
            for _, batch in enumerate(tqdm_dataloader): #eval loop
                input, label = batch
                output = assistant.test(input, label)
                tqdm_dataloader.set_description(f'\r[Epoch {epoch:2d}/{epochs}] {stats}')
                labels = labels + label.tolist()
                outputs = outputs + output.tolist()
            print('output shape', output.shape)
            nni.report_intermediate_result(stats.testing.accuracy*100)

            stats.update()

            if stats.testing.best_accuracy:
                count = 0
                labels = np.array(labels).flatten()
                outputs = np.array(outputs)
                predictions = compute_output_labels(outputs)
                gen_confusion_matrix(predictions,labels, trained_folder)
                torch.save(net.state_dict(), trained_folder + '/network.pt')
                stats.save(trained_folder + '/')
                net.grad_flow(trained_folder + '/')

            del outputs
            del labels



 
    
    nni.report_final_result(stats.testing.max_accuracy*100)
    stats.plot(figsize=(15, 5),path=f'{trained_folder}/')
    
    
    net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    net.export_hdf5(trained_folder + '/network.net')

    output, _ = net(input.to(device))
    print('output before image printing', output.shape)
    for i in range(5):
        try:
            inp_event = slayer.io.tensor_to_event(input[i].cpu().data.numpy().reshape(1, 6, -1 ))
            out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 7, -1))
            inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=10)
            out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=10)
            inp_anim.save(f'./gifs/inp{i}_label{label[i]}.gif', animation.PillowWriter(fps=30), dpi=300)
            out_anim.save(f'./gifs/out{i}_label{label[i]}.gif', animation.PillowWriter(fps=30), dpi=300)
        except:
            pass

if __name__ == '__main__':
    main()
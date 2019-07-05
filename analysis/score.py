#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
import os
from sys import argv
import numpy as np

import yaml
def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# Default I/O directories:
root_dir = "/Users/isabelleguyon/Documents/Projects/ParisSaclay/Projects/ChaLab/Examples/iris/"
default_input_dir = root_dir + "scoring_input_1_2"
default_output_dir = root_dir + "scoring_output"

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 1.0

metric_name = 'LHCb_CaloGAN_regression_metric'

ParticleMomentum_MEAN = np.array([0., 0.])
ParticlePoint_MEAN = np.array([0., 0.])

from prd_score import compute_prd_from_embedding
from calogan_metrics import get_assymetry, get_shower_width, get_sparsity_level
from calogan_metrics import get_physical_stats
from sklearn.metrics import auc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.batchnorm0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 16, 2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 2, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 2, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2)
        
        self.dropout = nn.Dropout(p=0.3)
        
        self.fc1 = nn.Linear(256, 256) 
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2 + 3)
        self.fc5 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.batchnorm0(self.dropout(x))
        x = self.batchnorm1(self.dropout(F.relu(self.conv1(x))))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.batchnorm3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x)) # 64, 5, 5
        x = x.view(len(x), -1)
        x = self.dropout(x)
        x = self.batchnorm4(self.dropout(F.relu(self.fc1(x))))
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.fc4(x), self.fc5(x)
    
    def get_encoding(self, x):
        x = self.batchnorm0(self.dropout(x))
        x = self.batchnorm1(self.dropout(F.relu(self.conv1(x))))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.batchnorm3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x)) # 64, 5, 5
        x = x.view(len(x), -1)
        x = self.dropout(x)
        x = self.batchnorm4(self.dropout(F.relu(self.fc1(x))))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def load_embedder(path):
    current_basepath = os.path.dirname(os.path.abspath(__file__))
    embedder = torch.load(os.path.join(current_basepath, path))
    embedder.eval()
    return embedder


def scoring_function(solution_file, predict_file):
    np.random.seed(1337)
    score = 0.
    
    solution = np.load(solution_file, allow_pickle=True)
    predict = np.load(predict_file, allow_pickle=True)
    
    EnergyDeposit_sol = solution['EnergyDeposit']
    ParticleMomentum_sol = solution['ParticleMomentum']
    ParticlePoint_sol = solution['ParticlePoint']
    
    EnergyDeposit_pred = predict['EnergyDeposit']
    ParticleMomentum_pred = solution['ParticleMomentum']
    ParticlePoint_pred = solution['ParticlePoint']
    
    embedder = load_embedder('./embedder.tp')
    EnergyDeposit_sol_emb = embedder.get_encoding(
        torch.tensor(EnergyDeposit_sol).float().view(-1, 1, 30, 30)
    ).detach().numpy()

    EnergyDeposit_pred_emb = embedder.get_encoding(
        torch.tensor(EnergyDeposit_sol).float().view(-1, 1, 30, 30)
    ).detach().numpy()
    
    precision, recall = compute_prd_from_embedding(
                        EnergyDeposit_sol_emb.reshape(len(EnergyDeposit_sol), -1), 
                        EnergyDeposit_pred_emb.reshape(len(EnergyDeposit_sol), -1),
                        num_clusters=100,
                        num_runs=100)
    
    auc_img = auc(precision, recall)
    
    physical_metrics_sol = get_physical_stats(
        EnergyDeposit_sol, 
        ParticleMomentum_sol,
        ParticlePoint_sol)

    physical_metrics_pred = get_physical_stats(
        EnergyDeposit_pred, 
        ParticleMomentum_pred,
        ParticlePoint_pred)
    
    precision, recall = compute_prd_from_embedding(
        physical_metrics_sol, 
        physical_metrics_pred,
        num_clusters=100,
        num_runs=100)
    
    auc_physical_metrics = auc(precision, recall)
    
    return min(auc_img, auc_physical_metrics)
# =============================== MAIN ========================================

if __name__ == "__main__":

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        # Create the output directory, if it does not already exist and open output files
    mkdir(output_dir)
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'wb')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'wb')

    # Get all the solution files from the solution directory
    solution_file = os.path.join(input_dir, 'ref', 'data_val_solution.npz')
    prediction_file = os.path.join(input_dir, 'res', 'data_val_prediction.npz')

    prediction_name = 'data_val_prediction'
    set_num = 1
    score_name = 'set%s_score' % set_num


    solution_file_test = os.path.join(input_dir, 'ref', 'data_test_solution.npz')
    prediction_file_test = os.path.join(input_dir, 'res', 'data_test_prediction.npz')

    scoring_function(solution_file_test, prediction_file_test)
    score = scoring_function(solution_file, prediction_file)
    print("======= Set %d" % set_num + " (" + prediction_name.capitalize() + "): score(" + score_name + ")=%0.12f =======" % score)
    html_file.write(("======= Set %d" % set_num + " (" + prediction_name.capitalize() + "): score(" + score_name + ")=%0.12f =======\n" % score).encode())

    # Write score corresponding to selected task and metric to the output file
    score_file.write((score_name + ": %0.12f\n" % score).encode())

    # Read the execution time and add it to the scores:
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        score_file.write("Duration: %0.6f\n" % metadata['elapsedTime'])
    except:
        score_file.write(b"Duration: 0\n")

        html_file.close()
    score_file.close()


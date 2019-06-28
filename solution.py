#!/usr/bin/python
import sys
import numpy as np

def main():
    input_dir, output_dir = sys.argv[1:]
    
    data_val = np.load(input_dir + '/data_val.npz', allow_pickle=True)
    N_val = len(data_val['ParticleMomentum'])

    data_test = np.load(input_dir + '/data_test.npz', allow_pickle=True)
    N_test = len(data_test['ParticleMomentum'])

    np.savez_compressed(output_dir + '/data_val_solution.npz', 
                         EnergyDeposit=np.random.randn(N_val, 30, 30))

    np.savez_compressed(output_dir + '/data_val_solution.npz',
                         EnergyDeposit=np.random.randn(N_test, 30, 30))

    return 0

if __name__ == "__main__":
    main()

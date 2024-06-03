import sys
import numpy as np

def txt_to_bin(txt_file, bin_file):
    data = np.loadtxt(txt_file, delimiter="\n", dtype=np.float32)
    new_data = data.astype(np.int8)
    new_data.tofile(bin_file)

txt_file = './dumper_tensor/root.conv1.Conv2dInt.q_input'
bin_file = './dumper_tensor/root.conv1.Conv2dInt.q_input.bin'

txt_to_bin(txt_file,bin_file)

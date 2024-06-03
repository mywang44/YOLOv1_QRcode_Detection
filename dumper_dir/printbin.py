
import numpy as np

def read_bin_and_write_to_txt(bin_file_path, txt_file_path):
    # 使用numpy读取二进制文件
    data = np.fromfile(bin_file_path, dtype=np.float32)

    # 将数据写入文本文件
    with open(txt_file_path, 'w') as txt_file:
        for value in data:
            txt_file.write(f"{value}\n")

if __name__ == "__main__":
    bin_file_path = './dumper_tensor/output.bin'
    txt_file_path = './dumper_tensor/output.txt'
    read_bin_and_write_to_txt(bin_file_path, txt_file_path)

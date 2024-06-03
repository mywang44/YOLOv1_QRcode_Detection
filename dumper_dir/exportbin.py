import cv2
import numpy as np
import struct

def save_image_as_cvmat_rgb(image_path, output_file):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image {image_path}.")
        return

    rows, cols, channels = image.shape

    with open(output_file, 'wb') as f:
        # 写入尺寸信息：先写入rows，然后cols，最后channels
        f.write(struct.pack('iii', rows, cols, channels))
        
        for row in image:
            f.write(row.tobytes())

    print(f"Saved {image_path} as CvMat-like format with RGB channels to {output_file}.")
    print(image.shape)

if __name__ == "__main__":
    save_image_as_cvmat_rgb("0000964.jpg", "./test_thinker/0000964.bin")
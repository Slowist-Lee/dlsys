from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION

        self.transforms = transforms

        with gzip.open(image_filename,'rb') as f_image:
            image=f_image.read(16)
            _,self.num_image,self.num_row,self.num_col=struct.unpack('>4I', image)
            num_pixels=self.num_image*self.num_row*self.num_col
            image_data=f_image.read(num_pixels)
            self.image = np.frombuffer(image_data, dtype=np.uint8).reshape(self.num_image,self.num_row*self.num_col).astype(np.float32)/255.0

        with gzip.open(label_filename,'rb') as f_label:
            label=f_label.read(8)
            _,self.num_label=struct.unpack('>2I', label)
            label_data=f_label.read(self.num_label)
            self.label = np.frombuffer(label_data, dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img=self.image[index].reshape(self.num_row,self.num_col,-1)
        label=self.label[index]
        if self.transforms:
            for transform in self.transforms:
                img=transform(img)
            # img实际上是一个batch的图片
        img=img.reshape(-1,self.num_row*self.num_col)
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.num_image
        ### END YOUR SOLUTION
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct,gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.images, self.labels = parse_mnist(image_filename=image_filename, label_filename=label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X, y = self.images[index], self.labels[index]
        if self.transforms:
            X = self.apply_transforms(X.reshape(28,28,-1))
            X = X.reshape(-1, 28*28)
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    magic number: 0x0802

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename) as image:
        magic, pictures, rows, cols = struct.unpack(">4i",image.read(16))
        assert(magic == 0x0803)
        pixels = rows * cols
        X = np.vstack([np.array(struct.unpack(f"{pixels}B", image.read(pixels)), dtype=np.float32) for _ in range(pictures)])
        X -= X.min()
        X /= X.max()

    with gzip.open(label_filename) as label:
        magic, labels = struct.unpack(">2i", label.read(8))
        assert(magic == 0x0801)
        y = np.array(struct.unpack(f"{labels}B", label.read()), dtype = np.uint8)

    return X, y
    ### END YOUR CODE

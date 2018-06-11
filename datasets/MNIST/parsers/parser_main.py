import os
import functools
import operator
import gzip
import struct
import array
import tempfile
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve  # py2
try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin
import numpy as np


# the url can be changed by the users of the library (not a constant)
datasets_url = 'http://yann.lecun.com/exdb/mnist/'
dataset_dir = 'C:\\Users\\sbhalla\\Documents\\git-repos\\datasets\\datasets\\MNIST\\'


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass

class mnist():
    def __init__(self):
        pass

    def download_file(self, fname, target_dir=None, force=False):
        """Download fname from the datasets_url, and save it to target_dir,
        unless the file already exists, and force is False.

        Parameters
        ----------
        fname : str
            Name of the file to download

        target_dir : str
            Directory where to store the file

        force : bool
            Force downloading the file, if it already exists

        Returns
        -------
        fname : str
            Full path of the downloaded file
        """
        if not target_dir:
            target_dir = tempfile.gettempdir()
        target_fname = os.path.join(target_dir, fname)

        if force or not os.path.isfile(target_fname):
            url = urljoin(datasets_url, fname)
            urlretrieve(url, target_fname)

        return target_fname


    def parse_idx(self, fd):
        """Parse an IDX file, and return it as a numpy array.

        Parameters
        ----------
        fd : file
            File descriptor of the IDX file to parse

        endian : str
            Byte order of the IDX file. See [1] for available options

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file

        1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
        """
        DATA_TYPES = {0x08: 'B',  # unsigned byte
                      0x09: 'b',  # signed byte
                      0x0b: 'h',  # short (2 bytes)
                      0x0c: 'i',  # int (4 bytes)
                      0x0d: 'f',  # float (4 bytes)
                      0x0e: 'd'}  # double (8 bytes)

        header = fd.read(4)
        if len(header) != 4:
            raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

        if zeros != 0:
            raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                                 'Found 0x%02x' % zeros)

        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        fd.read(4 * num_dimensions))

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(operator.mul, dimension_sizes)
        if len(data) != expected_items:
            raise IdxDecodeError('IDX file has wrong number of items. '
                                 'Expected: %d. Found: %d' % (expected_items, len(data)))

        return np.array(data).reshape(dimension_sizes)

    def parse_mnist_file(self, target_file):
        """Use the IDX file provided
        and return it as a numpy array.

        Parameters
        ----------
        target_file : str
            File to unzip and parse the file

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file
        """
        fopen = gzip.open if os.path.splitext(target_file)[1] == '.gz' else open
        with fopen(target_file, 'rb') as fd:
            return self.parse_idx(fd)

    def download_and_parse_mnist_file(self, fname, target_dir=None, force=False):
        """Download the IDX file named fname from the URL specified in dataset_url
        and return it as a numpy array.

        Parameters
        ----------
        fname : str
            File name to download and parse

        target_dir : str
            Directory where to store the file

        force : bool
            Force downloading the file, if it already exists

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file
        """
        fname = self.download_file(fname, target_dir=target_dir, force=force)
        fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
        with fopen(fname, 'rb') as fd:
            return self.parse_idx(fd)


    def train_images(self, download=True):
        """Return train images from Yann LeCun MNIST database as a numpy array.
        Download the file, if not already found in the temporary directory of
        the system.

        Returns
        -------
        train_images : numpy.ndarray
            Numpy array with the images in the train MNIST database. The first
            dimension indexes each sample, while the other two index rows and
            columns of the image
        """
        if download: return self.download_and_parse_mnist_file('train-images-idx3-ubyte.gz')
        else: return self.parse_mnist_file(dataset_dir+'train-images-idx3-ubyte.gz')


    def test_images(self, download=True):
        """Return test images from Yann LeCun MNIST database as a numpy array.
        Download the file, if not already found in the temporary directory of
        the system.

        Returns
        -------
        test_images : numpy.ndarray
            Numpy array with the images in the train MNIST database. The first
            dimension indexes each sample, while the other two index rows and
            columns of the image
        """
        if download: return self.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')
        else: return self.parse_mnist_file(dataset_dir+'t10k-images-idx3-ubyte.gz')


    def train_labels(self, download=True):
        """Return train labels from Yann LeCun MNIST database as a numpy array.
        Download the file, if not already found in the temporary directory of
        the system.

        Returns
        -------
        train_labels : numpy.ndarray
            Numpy array with the labels 0 to 9 in the train MNIST database.
        """
        if download: return self.download_and_parse_mnist_file('train-labels-idx1-ubyte.gz')
        else: return self.parse_mnist_file(dataset_dir+'train-labels-idx1-ubyte.gz')


    def test_labels(self, download=True):
        """Return test labels from Yann LeCun MNIST database as a numpy array.
        Download the file, if not already found in the temporary directory of
        the system.

        Returns
        -------
        test_labels : numpy.ndarray
            Numpy array with the labels 0 to 9 in the train MNIST database.
        """
        if download: return self.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')
        else: return self.parse_mnist_file(dataset_dir+'t10k-labels-idx1-ubyte.gz')


    def get_mnist_data(self):
        """Return the train images and labels + test images and labels.

        Returns
        -------
        x_train: numpy.ndarray
            Numpy array with images in train MNIST database.
        y_train: numpy.ndarray
            Numpy array with labels in train MNIST database.
        x_test, y_test
        """
        return (self.train_images(download=False),self.train_labels(download=False),\
            self.test_images(download=False), self.test_labels(download=False))
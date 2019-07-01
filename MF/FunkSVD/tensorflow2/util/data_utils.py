# -*- coding: utf-8 -*-

import numpy as np

def get_zip_file1(url, filepath):
    from io import BytesIO
    from io import StringIO
    from zipfile import ZipFile
    import requests
    
    
    file_content = requests.get(url).content
    print(file_content[:10], type(file_content))
    byteFile = BytesIO(file_content)
    print(type(byteFile))
    zipfile = ZipFile(byteFile)
    print('zipfile', type(zipfile))
    file = zipfile.open(filepath).read().decode('utf8')
    print('fiel', type(file))
    ret = StringIO(file)
    print('ret', type(ret))
    return ret

def get_zip_file(url, filepath):
    """Gets zip file from url.
    Args:
        url: A string, the url of zip file.
        filepath: A string, the file path inside the zip file.
    Returns:
        A String, the content of wanted file.
    """

    from io import BytesIO
    from io import StringIO
    from zipfile import ZipFile
    import requests

    zipfile = ZipFile(BytesIO(requests.get(url).content))
    file = zipfile.open(filepath).read().decode('utf8')

    return StringIO(file)


class BatchGenerator(object):
    def __init__(self, x, y=None, batch_size=1024, shuffle=True):
        if y is not None and x.shape[0] != y.shape[0]:
            raise ValueError('The shape 0 of x should be equal to that of y')
        self.x = x
        self.y = y
        self.length = x.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def next(self):
        start = end = 0
        length = self.length
        batch_size = self.batch_size
        
        if self.shuffle:
            permutation = np.random.permutation(length)
            self.x = self.x[permutation]
            self.y= self.y[permutation]
        
        flag = False
        while not flag:
            end += batch_size
            
            if (end+batch_size > length):
                end = length-1
                flag = True
                
            yield self.get_batch(start, end)
            start = end
            
    def get_batch(self, start, end):
        if(self.y is not None):
            return self.x[start:end], self.y[start:end]
        else:
            return self.x[start:end]
                

if(__name__ == "__main__"):
    URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    FILE_PATH = 'ml-1m/ratings.dat'
    ans = get_zip_file1(URL, FILE_PATH)
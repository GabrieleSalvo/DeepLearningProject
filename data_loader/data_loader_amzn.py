from torchvision import datasets, transforms
from base import BaseDataLoader
import pandas as pd
import gzip

class AmznDataLoader(BaseDataLoader):
    """
    AMZN data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        df = self.getDF('./data/reviews_Amazon_Instant_Video_5.json.gz')
        display(df)
    
    def parse(self, path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)
        
    def getDF(self, path):
        i = 0
        df = {}
        for d in self.parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')
"""
AIO -- All Trains in One
"""

from trains.multiTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
           'tsafn': TsAFN,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName.lower()](args)

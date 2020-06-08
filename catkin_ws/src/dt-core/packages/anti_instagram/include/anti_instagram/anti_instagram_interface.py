import duckietown_utils as dtu
from abc import ABCMeta, abstractmethod

__all__ = [
    'AntiInstagramInterface',
]

class AntiInstagramInterface(object):
    
    FAMILY = 'anti_instagram'
    
    __metaclass__ = ABCMeta
   
    
    @abstractmethod
    @dtu.contract(bgr='array[HxWx3](uint8)', returns='None')
    def calculate_color_balance_thresholds(self, bgr, scale=0.2, percentage=0.8):
        """ Computes the transform """

    @abstractmethod
    @dtu.contract(bgr='array[HxWx3](uint8)', returns='None')
    def apply_color_balance(self, bgr, scale=0.4):
        """ Applies the transform """
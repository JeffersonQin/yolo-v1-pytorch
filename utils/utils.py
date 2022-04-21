import numpy as np
import time


__all__ = ['Accumulator', 'Timer']


# from d2l
class Accumulator(object):
    """
    Sum a list of numbers over time
    from: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/base.py
    """
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0] * len(self.data)
    def __getitem__(self, i):
        return self.data[i]


class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer and record the time in a list"""
        self.times.append(time.time() - self.start_time)
        return self.times[-1]
        
    def avg(self):
        """Return the average time"""
        return sum(self.times)/len(self.times)
    
    def sum(self):
        """Return the sum of time"""
        return sum(self.times)
        
    def cumsum(self):
        """Return the accumuated times"""
        return np.array(self.times).cumsum().tolist()
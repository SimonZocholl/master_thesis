import numpy as np

class MyTransform:
    def __init__(self, dataset, a,b) -> None:
        self.mi = np.min(dataset.features, axis=(0,1,-1))
        self.mx = np.max(dataset.features, axis=(0,1,-1))
        
        self.a = a
        self.b = b
        self.eps =  1e-7

    def min_max_scale(self, x):
        mit = np.tile(self.mi, (x.shape[-1],1)).T
        mxt = np.tile(self.mx, (x.shape[-1],1)).T
        x_prime = ((self.a + (x- mit)*(self.b -self.a))+ self.eps) / ((mxt -mit)+ self.eps)
        return x_prime
    
    def inv_min_max_scale(self,x_prime):
        mit = np.tile(self.mi, (x_prime.shape[-1],1)).T
        mxt = np.tile(self.mx, (x_prime.shape[-1],1)).T

        x = (((x_prime - self.a) * (mxt - mit))+ self.eps)/((self.b - self.a)+ self.eps) + mit
        return x
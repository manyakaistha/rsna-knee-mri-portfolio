"""Tie-correct rank AUC without SciPy binary dependencies."""
import numpy as np

def ranks(values):
    values=np.asarray(values);order=np.argsort(values,kind='stable');sorted_values=values[order]
    starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1];ends=np.r_[starts[1:],len(values)]
    result=np.empty(len(values),float)
    for lo,hi in zip(starts,ends):result[order[lo:hi]]=(lo+hi-1)/2+1
    return result

def binary_auc(y,p):
    y=np.asarray(y,dtype=bool);n=int(y.sum());m=len(y)-n
    if not n or not m:raise ValueError('AUC requires two classes')
    return float((ranks(p)[y].sum()-n*(n+1)/2)/(n*m))

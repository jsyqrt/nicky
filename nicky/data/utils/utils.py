# coding: utf-8
# utils.py

import pickle

__all__ = ['save_obj', 'load_obj']

def save_obj(obj, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, -1)
        state = True
    except:
        state = False
    print(state, 'with save %s to %s' %(obj, path))
    return state

def load_obj(path):
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except:
        obj = False
    return obj

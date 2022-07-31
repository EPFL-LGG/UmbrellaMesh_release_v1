import sys
sys.path.append('..')
import numpy as np
import numpy.linalg as la
from test_constructor import construct_umbrella

import pickle 
import gzip

curr_um = construct_umbrella('../../data/sphere_cap_0.3.json.gz')
curr_um.attractionWeight = 20000

def test_pickle_joint():
    pickle.dump(curr_um.joint(0), gzip.open('test_pickle_joint.pkl.gz', 'w'))
    pickle.load(gzip.open('test_pickle_joint.pkl.gz', 'r'))

def test_pickle_seg():
    pickle.dump(curr_um.segment(0), gzip.open('test_pickle_segment.pkl.gz', 'w'))
    pickle.load(gzip.open('test_pickle_segment.pkl.gz', 'r'))

def test_pickle_tsf():
    pickle.dump(curr_um.getTargetSurface(), gzip.open('test_pickle_tsf.pkl.gz', 'w'))
    pickle.load(gzip.open('test_pickle_tsf.pkl.gz', 'r'))

def test_pickle_um():
    pickle.dump(curr_um, gzip.open('test_pickle_um.pkl.gz', 'w'))
    load_um = pickle.load(gzip.open('test_pickle_um.pkl.gz', 'r'))
    assert load_um.attractionWeight == curr_um.attractionWeight   

if __name__ == "__main__":
    test_pickle_joint()
    test_pickle_seg()
    test_pickle_tsf()
    test_pickle_um()
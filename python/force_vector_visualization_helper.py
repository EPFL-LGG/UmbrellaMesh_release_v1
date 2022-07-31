elastic_rods_dir = '../elastic_rods/python/'
weaving_dir = './'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir); sys.path.append(weaving_dir)
import numpy as np, elastic_rods, linkage_vis
import numpy.linalg as la
from elastic_rods import EnergyType, InterleavingType
import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1;
OPTS.beta = 1e-8
OPTS.niter = 200
OPTS.verboseNonPosDef = False
rw = 1
sw = 10
drw = 0.1
dsw = 0.1
import pickle 
import gzip
import matplotlib.pyplot as plt

class ForceVector:
    '''
    A data class for gathering information needed for rendering force vectors on linkages. 
    '''
    def __init__(self, joint_pos, force_vector):
        self.joint_pos = joint_pos
        self.magnitude = la.norm(force_vector)
        self.direction = force_vector / self.magnitude
        self.color = None

def get_color_scheme(colors):
    '''
    Color scheme for force vectors.
    '''
    cmap = plt.cm.plasma
    # cmap = plt.cm.PuRd
    return cmap(colors)
#     import proplot as plot

#     # Colormap from named color
#     # The trailing '_r' makes the colormap go dark-to-light instead of light-to-dark
#     cmap1 = plot.Colormap('violet red', name='pacific', fade=100, space='hsl')
#     # The color map has 256 colors.
#     colors = np.round(colors * 256)
#     return plot.to_rgb(cmap1(colors), space = 'hsl')

def get_force_vectors(linkage_pickle_name, omitBoundary = True):
    '''
    Compute the separation forces of a given linkage and return of list of force vector objects.
    '''
    # Create linkage object from pickle.
    linkage = pickle.load(gzip.open(linkage_pickle_name, 'r'))

    # Compute forces.
    AForceOnJoint = linkage.UmbrellaRivetNetForceAndTorques()[:, 0:3]
    

    separationForceVectors = []
    epsilon = 1e-15
    for udi in range(len(AForceOnJoint)):
        f = AForceOnJoint[udi]
        ji = linkage.getUmbrellaCenterJi(int(udi / 2), udi % 2)  
        j = linkage.joint(ji)
        separationDirection = j.ghost_normal()
        separationForce = np.clip(f.dot(separationDirection), 0, None) * separationDirection
        if la.norm(separationForce) > epsilon:
            separationForceVectors.append(ForceVector(j.position, separationForce))
    return separationForceVectors

def write_force_vector_visualization_file(list_of_pickle, list_of_output_name):
    # Each element of this list is a pair of [separation_fv, tangential_fv].
    list_of_FV = [get_force_vectors(name) for name in list_of_pickle]

    # Get Max Range.
    max_separation = np.amax(np.hstack([[fv.magnitude for fv in enum_fv] for enum_fv in list_of_FV]))
    print(max_separation)
    # Compute Color.
    def set_color(fv, norm):
        fv.color = get_color_scheme(fv.magnitude / norm)

    [[set_color(fv, max_separation) for fv in enum_fv] for enum_fv in list_of_FV]

    # Normalize Force Magnitude to be Max.
    def normalize_magnitude(fv, norm):
        fv.magnitude = fv.magnitude / norm 

    [[normalize_magnitude(fv, max_separation) for fv in enum_fv] for enum_fv in list_of_FV]
  
    # Write To File.
    def write_to_file(name, fv_list):
        with open(name, 'w') as f:
            for fv in fv_list:
                f.write('{}, {}, {}, {}\n'.format(tuple(fv.direction), tuple(fv.joint_pos), fv.magnitude, fv.color))

    [write_to_file('{}_separationForceVectors.txt'.format(list_of_output_name[i]), list_of_FV[i]) for i in range(len(list_of_FV))]

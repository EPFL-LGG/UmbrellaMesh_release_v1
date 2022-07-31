#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('..')
import umbrella_mesh
import elastic_rods
import linkage_vis
from umbrella_mesh import UmbrellaEnergyType
from bending_validation import suppress_stdout as so
from visualization_helper import *

import pipeline_helper, importlib, design_optimization_analysis
with so(): importlib.reload(pipeline_helper)
with so(): importlib.reload(design_optimization_analysis)

from pipeline_helper import UmbrellaOptimizationCallback, allEnergies, allGradientNorms, allDesignObjectives, allDesignGradientNorms, set_joint_vector_field, show_center_joint_normal, show_joint_normal

from design_optimization_analysis import DesignOptimizationAnalysis

import umbrella_optimization
import umbrella_optimization_finite_diff
from umbrella_optimization import OptEnergyType

import numpy as np
import numpy.linalg as la

import pickle, gzip

from configuration import *

import parallelism
parallelism.set_max_num_tbb_threads(24)
parallelism.set_hessian_assembly_num_threads(8)
parallelism.set_gradient_assembly_num_threads(8)


# ### Initialization

# In[2]:


name = 'hemisphere_5t'
input_path = '../../data/{}.json.gz'.format(name)

io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path, handleBoundary=False, handlePivots = True)


# curr_um = pickle.load(gzip.open('../../hive_output/hive_optimized_equilibrium_2022_01_25_11_33_target_height_factor_5.0.pkl.gz', 'r'))


# In[3]:


# curr_um = pickle.load(gzip.open('../../Optimized_model/tigridia/Copy of tigridia_optimized_equilibrium_2022_01_23_16_08_target_height_factor_5.0.pkl.gz', 'r'))


# #### Pin Rigid Motion

# In[4]:


use_pin = False

driver = curr_um.centralJoint()
jdo = curr_um.dofOffsetForJoint(driver)
fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()


# In[5]:


import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1
OPTS.beta = 1e-6
OPTS.niter = 300
OPTS.verboseNonPosDef = False

rod_colors = get_color_field(curr_um, input_data)

# lview = linkage_vis.LinkageViewer(curr_um, width=1024, height=600)
# lview.update(scalarField = rod_colors)
# lview.show()


from equilibrium_solve_analysis import EquilibriumSolveAnalysis
eqays = EquilibriumSolveAnalysis(curr_um)

curr_um = pickle.load(gzip.open('output/hemisphere_5t/2022_04_30_16_45/target_height_factor_5.0.pkl.gz', 'r'))


# In[10]:



import py_newton_optimizer
opt_opts = py_newton_optimizer.NewtonOptimizerOptions()
opt_opts.gradTol = 1e-8
opt_opts.verbose = 10
opt_opts.beta = 1e-6
opt_opts.niter = 50
opt_opts.verboseNonPosDef = False


# In[11]:


rest_height_optimizer = umbrella_optimization.UmbrellaRestHeightsOptimization(umbrella_optimization.UmbrellaOptimization(curr_um, opt_opts, 2.5, -1, False, fixedVars))

rest_height_optimizer.beta = 0
rest_height_optimizer.gamma = 1
rest_height_optimizer.eta = 0
rest_height_optimizer.zeta = 0# 1e1
rest_height_optimizer.iota = 0


# In[ ]:


allDesignObjectives(rest_height_optimizer)

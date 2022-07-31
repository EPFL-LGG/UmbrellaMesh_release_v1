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

from datetime import datetime

import os

from load_jsondata import read_data, write_deformed_config
import mesh
import importlib, pipeline_helper

from equilibrium_solve_analysis import EquilibriumSolveAnalysis
import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1
OPTS.beta = 1e-6
OPTS.niter = 300
OPTS.verboseNonPosDef = False

import time


import umbrella_optimization

import py_newton_optimizer
opt_opts = py_newton_optimizer.NewtonOptimizerOptions()
opt_opts.gradTol = 1e-8
opt_opts.verbose = 10
opt_opts.beta = 1e-6
opt_opts.niter = 600
opt_opts.verboseNonPosDef = False


import pickle 
import gzip

import compute_vibrational_modes

from configuration import *
import json

model_names = []
import os.path as ops

benchmark_data = {}
with open('umbrella_model_names.json') as f:
    data = json.load(f)
stats = []
for model_info in data['models']:
    # ### Initialization
    input_path = '../../data/{}.json.gz'.format(model_info['name'])
    io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path, handleBoundary=model_info['use_boundary'])
    curr_stats = {}
    curr_stats['name'] = model_info['name']
    curr_stats['num_umbrellas'] = curr_um.numUmbrellas()
    curr_stats['num_joints'] = curr_um.numJoints()
    curr_stats['num_segments'] = curr_um.numSegments()
    curr_stats['num_dofs'] = curr_um.numDoF()
    if model_info['optimized_parameters'] != '':
        params = json.load(gzip.open(model_info['optimized_parameters'], 'r'))
        m = params['plate_edge_length'] *  params['bbox_diagonal']
        ht = target_height_multiplier
        heights = np.array(params['optim_heights'])
        top_bottom_heights = heights.reshape(2, int(len(heights) / 2))
        h = (np.max(np.min(top_bottom_heights, 0)) ** 2 - ht ** 2) ** 0.5
        max_scale_factor = (m + 2 * np.sqrt(3) * h) / m
        curr_stats['max_scale_factor'] = max_scale_factor

        min_h = (np.min(np.min(top_bottom_heights, 0)) ** 2 - ht ** 2) ** 0.5
        min_scale_factor = (m + 2 * np.sqrt(3) * min_h) / m
        
        curr_stats['relative_scale_factor'] = max_scale_factor / min_scale_factor


    stats.append(curr_stats)
with open('benchmark_results.json', 'w') as f:
    json.dump(stats, f, indent = 4)
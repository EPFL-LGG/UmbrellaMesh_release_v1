#!/usr/bin/env python
# coding: utf-8


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

def generate_validation_images(PARL, output_name):
    io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path, handleBoundary=False, handlePivots = True)
    curr_um.setPerArmRestLength(PARL)

    # #### Pin Rigid Motion
    use_pin = True

    driver = curr_um.centralJoint()
    jdo = curr_um.dofOffsetForJoint(driver)
    fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()
    import py_newton_optimizer
    OPTS = py_newton_optimizer.NewtonOptimizerOptions()
    OPTS.gradTol = 1e-8
    OPTS.verbose = 1
    OPTS.beta = 1e-6
    OPTS.niter = 300
    OPTS.verboseNonPosDef = False

    rod_colors = get_color_field(curr_um, input_data)
    import mesh
    view = linkage_vis.LinkageViewerWithSurface(curr_um, target_mesh, width=1024, height=600)
    set_surface_view_options(view, color = 'green', surface_color = 'gray', umbrella_transparent = False, surface_transparent = True)
    view.averagedMaterialFrames = True
    view.showScalarField(rod_colors)
    view.show()
    def eqm_callback(prob, i):
        if (i % 10 == 0):
            view.showScalarField(rod_colors)
    # ### Offcreen Render
    width = 1024
    height = 600

    import time, os
    time_stamp = time.strftime("%Y_%m_%d_%H_%M")


    import os
    output_folder = 'video_{}_{}'.format(time_stamp, output_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  
    # ### Undeployment

    use_pin = True

    driver = curr_um.centralJoint()
    jdo = curr_um.dofOffsetForJoint(driver)
    fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()


    for deployment_iter in range(100):
        io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path, handleBoundary=False, handlePivots = True)
        curr_um.setPerArmRestLength(PARL)
        while (True):
            results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)
            if results.success:
                break

        # ### Deployment
        configure_umbrella_pre_deployment(curr_um, thickness, target_height_multiplier)
        curr_um.attractionWeight = 0
        break_input_angle_symmetry(curr_um)
        insert_randomness(curr_um)

        results = staged_deployment(curr_um, np.logspace(-4, 0, 5), None, OPTS, fixedVars)
        view = linkage_vis.LinkageViewerWithSurface(curr_um, target_mesh, width=1024, height=600)
        set_surface_view_options(view, color = 'green', surface_color = 'gray', umbrella_transparent = False, surface_transparent = True)
        view.averagedMaterialFrames = True
        view.showScalarField(rod_colors)
        view.setCameraParams(((-3, -2, 0.05),
                              (0.05499289949481162, -0.06863987888969214, 0.996124664904531),
                              (0.0, 0.0, 0.0)))    
        render = view.offscreenRenderer(width, height)
        render.render()
        render.save("{}/{}.png".format(output_folder, deployment_iter))

curr_um = pickle.load(gzip.open("../../output/squidward_highres_optimized_equilibrium_2022_01_20_11_58_target_height_factor_5.0.pkl.gz", 'r'))
PARL = curr_um.getPerArmRestLength()

name = 'squidward_highres'
input_path = '../../data/{}.json.gz'.format(name)

generate_validation_images(PARL, name)

curr_um = pickle.load(gzip.open("../../output/squidward_highres_single_height_optimized_equilibrium_2022_01_20_00_11_target_height_factor_5.0.pkl.gz", 'r'))
PARL = curr_um.getPerArmRestLength()

name = 'squidward_highres'
input_path = '../../data/{}.json.gz'.format(name)

generate_validation_images(PARL, name + "single_height")
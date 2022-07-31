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


importlib.reload(pipeline_helper)

from pipeline_helper import set_joint_vector_field, show_center_joint_normal, show_joint_normal

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


import pipeline_helper, importlib, design_optimization_analysis
with so(): importlib.reload(pipeline_helper)
with so(): importlib.reload(design_optimization_analysis)

from pipeline_helper import UmbrellaOptimizationCallback

from umbrella_optimization import OptEnergyType

from design_optimization_analysis import DesignOptimizationAnalysis

import pickle 
import gzip

import compute_vibrational_modes

from configuration import *

camParams = ((20.200270957406428, -11.68904147568453, 6.1324571979764775),
 (-0.19887937946315074, 0.204946824965005, 0.9583547314857459),
 (3.3820736007969714, 3.3088268912141094, -0.5650191603439246))

def run_stiffness_analysis(output_folder_name, name):
    start_time = datetime.now()

    if not os.path.exists('{}/{}'.format(output_folder_name, name)):
        os.makedirs('{}/{}'.format(output_folder_name, name))  
    # ### Initialization
    input_path = '../../data/{}.json.gz'.format(name)
    io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path)


    with open ('{}/{}/{}_log_{}.txt'.format(output_folder_name, name, name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), 'w') as f:

        # #### Pin Rigid Motion
        # 
        # 
        use_pin = False

        driver = curr_um.centralJoint()
        jdo = curr_um.dofOffsetForJoint(driver)
        fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

        rod_colors = get_color_field(curr_um, input_data)

        view = linkage_vis.LinkageViewerWithSurface(curr_um, target_mesh, width=1024, height=600)
        set_surface_view_options(view, color = 'green', surface_color = 'gray', umbrella_transparent = False, surface_transparent = True)
        view.averagedMaterialFrames = True
        view.showScalarField(rod_colors)

        eqays = EquilibriumSolveAnalysis(curr_um)
        def eqm_callback(prob, i):
            eqays.record(prob)
            if (i % 2 == 0):
                view.showScalarField(rod_colors)
        
        configure_umbrella_pre_deployment(curr_um, thickness, target_height_multiplier)
        allGradientNorms(curr_um)

        break_input_angle_symmetry(curr_um)

        view.showScalarField(rod_colors)

        results = staged_deployment(curr_um, np.logspace(-3, 0, 4), eqm_callback, OPTS, fixedVars)


        # ### Initialize Design Optimization
        configure_umbrella_optimization(curr_um)


        with so(): results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = opt_opts, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

        opt_opts.niter = 50

        results.success

        optimizer = umbrella_optimization.UmbrellaOptimization(curr_um, opt_opts, 2.5, -1, False, fixedVars)

        optimizer.beta = 1 * 1e6
        optimizer.gamma = 1
        optimizer.eta = 0
        optimizer.zeta = 0# 1e1

        rest_height_optimizer = umbrella_optimization.UmbrellaRestHeightsOptimization(optimizer)
      
        doptays = DesignOptimizationAnalysis(rest_height_optimizer)
        def eqm_callback(prob, i):
            eqays.record(prob)
            if (i % 2 == 0):
                view.showScalarField(rod_colors)


        pipeline_helper.prev_time_stamp = time.time()

        uo = rest_height_optimizer.get_parent_opt()

        uo.equilibriumOptimizer.options.verbose = 1
        #uo.equilibriumOptimizer.options.verboseWorkingSet = True
        uo.equilibriumOptimizer.options.gradTol = 1e-10
        # Hold the closest points fixed in the target-attraction term of the equilibrium solve:
        # this seems to make the design optimization much more robust.
        uo.setHoldClosestPointsFixed(True, False)

        tfview = pipeline_helper.TargetFittingVisualization(curr_um, uo.target_surface_fitter, view)
        cb = pipeline_helper.UmbrellaOptimizationCallback(rest_height_optimizer, view, True, False, 1, rod_colors, doptays.record, tfview=tfview)

        algorithm = umbrella_optimization.OptAlgorithm.NEWTON_CG
        with so(): solverStatus = umbrella_optimization.optimize(rest_height_optimizer, algorithm, 10000, 0.005, 1e-5, cb, input_data["plate_edge_length"] / 30 * 32)
        f.write('design optimization solverStatus: {}\n'.format(solverStatus))

        doptays.plot()

        # ### Get true equilibrium state
        use_pin = True

        driver = curr_um.centralJoint()
        jdo = curr_um.dofOffsetForJoint(driver)
        fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

        configure_umbrella_true_equlibrium(curr_um, thickness, target_height_multiplier)


        allEnergies(curr_um)

        OPTS.niter = 500

        results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

        f.write('equilibrium solve status: {}\n'.format(results.success))


        pickle.dump(curr_um, gzip.open('{}/{}/{}_optimized_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder_name, name, name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), 'w'))

        write_deformed_config(curr_um, input_path, output_path = '{}/{}/{}_optimized_equilibrium_rendering_output_{}_target_height_factor_{}.json.gz'.format(output_folder_name, name, name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), write_stress = False, is_rest_state = False)

        renderToFile('{}/{}/{}_target_height_factor_{}_{}.png'.format(output_folder_name, name, name, target_height_multiplier, time.strftime("%Y_%m_%d_%H_%M")), view, renderCam=camParams)

        # ### Vibration Mode Analysis
        allEnergies(curr_um)


        class ModalAnalysisWrapper:
            def __init__(self, um):
                self.um = um
            def hessian(self):
                return self.um.hessian(umbrellaEnergyType=umbrella_mesh.UmbrellaEnergyType.Full)
            def massMatrix(self): return self.um.massMatrix()
            def lumpedMassMatrix(self): return self.um.lumpedMassMatrix()

        lambdas, modes = compute_vibrational_modes.compute_vibrational_modes(ModalAnalysisWrapper(curr_um), fixedVars=curr_um.rigidJointAngleDoFIndices(), mtype=compute_vibrational_modes.MassMatrixType.FULL, n=16, sigma=-1e-6)

        # Save lambdas
        np.save('{}/{}/lambdas_{}_target_height_factor_{}_{}'.format(output_folder_name, name, name, target_height_multiplier, time.strftime("%Y_%m_%d_%H_%M")), lambdas)
        np.save('{}/{}/modes_{}_target_height_factor_{}_{}'.format(output_folder_name, name, name, target_height_multiplier, time.strftime("%Y_%m_%d_%H_%M")), modes)

        f.write("The first seven eigenvalues are {}\n".format(lambdas[:7]))
        modeVector = modes[:, 6]
        paramVelocity = curr_um.approxLinfVelocity(modeVector)
        normalizedOffset = modeVector * (curr_um.characteristicLength() / paramVelocity)

        save_dof = curr_um.getDoFs()
        curr_um.setDoFs(save_dof + normalizedOffset * 5)
        pickle.dump(curr_um, gzip.open('{}/{}/{}_mode_plus_{}_target_height_factor_{}.pkl.gz'.format(output_folder_name, name, name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), 'w'))
        
        write_deformed_config(curr_um, input_path, output_path = '{}/{}/{}_mode_plus_rendering_output_{}_target_height_factor_{}.json.gz'.format(output_folder_name, name, name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), write_stress = False, is_rest_state = False)


        curr_um.setDoFs(save_dof - normalizedOffset * 5)
        pickle.dump(curr_um, gzip.open('{}/{}/{}_mode_minus_{}_target_height_factor_{}.pkl.gz'.format(output_folder_name, name, name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), 'w'))

        write_deformed_config(curr_um, input_path, output_path = '{}/{}/{}_mode_minus_rendering_output_{}_target_height_factor_{}.json.gz'.format(output_folder_name, name, name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), write_stress = False, is_rest_state = False)

        f.write("Runnning this experiment on {} takes {} seconds\n".format(name, (datetime.now() - start_time).total_seconds()))

output_folder_name = 'output_{}'.format(time.strftime("%Y_%m_%d_%H_%M"))
run_stiffness_analysis(output_folder_name, 'saddle_10t')
run_stiffness_analysis(output_folder_name, 'saddle_5t')
run_stiffness_analysis(output_folder_name, 'saddle_2t')
run_stiffness_analysis(output_folder_name, 'hemisphere_10t')
run_stiffness_analysis(output_folder_name, 'hemisphere_5t')
run_stiffness_analysis(output_folder_name, 'hemisphere_2t')


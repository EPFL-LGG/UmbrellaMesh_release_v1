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

from pipeline_helper import UmbrellaOptimizationCallback, allEnergies, allGradientNorms, allDesignObjectives, allDesignGradientNorms, save_data, set_joint_vector_field, show_center_joint_normal, show_joint_normal

from design_optimization_analysis import DesignOptimizationAnalysis

import umbrella_optimization
import umbrella_optimization_finite_diff
from umbrella_optimization import OptEnergyType

import numpy as np
import numpy.linalg as la

import pickle, gzip

from configuration import *
from datetime import datetime


import parallelism
parallelism.set_max_num_tbb_threads(24)
parallelism.set_hessian_assembly_num_threads(8)
parallelism.set_gradient_assembly_num_threads(8)



# ### Initialization
def run_ablation(name, handleBoundary):
    start_time = datetime.now()

    # name = 'hive'
    input_path = '../../data/{}.json.gz'.format(name)

    import time
    time_stamp = time.strftime("%Y_%m_%d_%H_%M")
    import os
    output_folder = '{}_{}'.format(time_stamp, name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  

    io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path, handleBoundary=handleBoundary, handlePivots = True)

    with open ('{}/{}_log_{}.txt'.format(output_folder, name, name, time_stamp, target_height_multiplier), 'w') as f:

        # #### Pin Rigid Motion

        use_pin = False

        driver = curr_um.centralJoint()
        jdo = curr_um.dofOffsetForJoint(driver)
        fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

        rod_colors = get_color_field(curr_um, input_data)


        import mesh
        view = linkage_vis.LinkageViewerWithSurface(curr_um, target_mesh, width=1024, height=600)
        set_surface_view_options(view, color = 'green', surface_color = 'gray', umbrella_transparent = False, surface_transparent = True)
        view.averagedMaterialFrames = True
        view.showScalarField(rod_colors)

        from equilibrium_solve_analysis import EquilibriumSolveAnalysis
        eqays = EquilibriumSolveAnalysis(curr_um)
        def eqm_callback(prob, i):
            eqays.record(prob)
            if (i % 2 == 0):
                view.showScalarField(rod_colors)

        import py_newton_optimizer
        OPTS = py_newton_optimizer.NewtonOptimizerOptions()
        OPTS.gradTol = 1e-8
        OPTS.verbose = 1
        OPTS.beta = 1e-6
        OPTS.niter = 300
        OPTS.verboseNonPosDef = False

        configure_umbrella_pre_deployment(curr_um, thickness, target_height_multiplier)

        allGradientNorms(curr_um)


        break_input_angle_symmetry(curr_um)

        view.showScalarField(rod_colors)

        results = staged_deployment(curr_um, np.logspace(-3, 0, 4), eqm_callback, OPTS, fixedVars)

        results.success

        # ### Get true equilibrium state

        curr_um.attractionWeight = 1e-7
        with so(): results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

        input_pickle_path = '{}/{}_input_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

        input_rendering_path = '{}/{}_input_equilibrium_{}_rendering_output_{}.json.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

        save_data(curr_um, input_pickle_path, input_rendering_path, input_path, False, handleBoundary)

        # ### Initialize Design Optimization
        configure_umbrella_optimization(curr_um, bdryMultiplier = 1)

        import py_newton_optimizer
        opt_opts = py_newton_optimizer.NewtonOptimizerOptions()
        opt_opts.gradTol = 1e-8
        opt_opts.verbose = 10
        opt_opts.beta = 1e-6
        opt_opts.niter = 600
        opt_opts.verboseNonPosDef = False

        results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = opt_opts, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

        opt_opts.niter = 50

        results.success

        # Run target surface fitting first

        optimizer = umbrella_optimization.UmbrellaOptimization(curr_um, opt_opts, 2.5, -1, False, fixedVars)

        optimizer.beta = 1 * 1e6
        optimizer.gamma = 1
        optimizer.eta = 0
        optimizer.zeta = 0# 1e1
        optimizer.iota = 0

        rest_height_optimizer = umbrella_optimization.UmbrellaRestHeightsOptimization(optimizer)
        single_rest_height_optimizer = umbrella_optimization.UmbrellaSingleRestHeightOptimization(rest_height_optimizer)

        rest_height_optimizer.newPt(rest_height_optimizer.params())

        doptays = DesignOptimizationAnalysis(rest_height_optimizer)

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
        #algorithm = umbrella_optimization.OptAlgorithm.BFGS
        arm_length_lower_bound = input_data["plate_edge_length"] / 30 * 32

        solverStatus = umbrella_optimization.optimize(rest_height_optimizer, algorithm, 1000, 0.005, 1e-5, cb, arm_length_lower_bound)

        curr_um.attractionWeight = 1e-7
        solverStatus = umbrella_optimization.optimize(rest_height_optimizer, algorithm, 1000, 0.005, 1e-5, cb, arm_length_lower_bound)

        f.write("Solver status for tsf: {} \n".format(solverStatus))

        tsf_pickle_path = '{}/{}_tsf_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

        tsf_rendering_path = '{}/{}_tsf_equilibrium_{}_rendering_output_{}.json.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

        save_data(curr_um, tsf_pickle_path, tsf_rendering_path, input_path, False, handleBoundary)


        # Then run force optimization

        optimizer.beta = 1 * 1e6
        optimizer.gamma = 1
        optimizer.eta = 0
        optimizer.zeta = 0# 1e1
        optimizer.iota = 1e10

        import force_analysis
        cfm = force_analysis.UmbrellaForceMagnitudes(curr_um)

        normalActivationThreshold = min(np.percentile(cfm[:, 0], 30), 0)
        f.write('normalActivationThreshold: {}\n'.format(normalActivationThreshold))

        optimizer.objective.terms[-1].term.normalActivationThreshold = normalActivationThreshold

        optimizer.objective.terms[-1].term.normalWeight = 1
        optimizer.objective.terms[-1].term.tangentialWeight = 0
        optimizer.objective.terms[-1].term.torqueWeight = 0

        rest_height_optimizer = umbrella_optimization.UmbrellaRestHeightsOptimization(optimizer)
        single_rest_height_optimizer = umbrella_optimization.UmbrellaSingleRestHeightOptimization(rest_height_optimizer)

        rest_height_optimizer.newPt(rest_height_optimizer.params())

        doptays = DesignOptimizationAnalysis(rest_height_optimizer)

        import time
        pipeline_helper.prev_time_stamp = time.time()

        uo = rest_height_optimizer.get_parent_opt()

        uo.equilibriumOptimizer.options.verbose = 1
        #uo.equilibriumOptimizer.options.verboseWorkingSet = True
        uo.equilibriumOptimizer.options.gradTol = 1e-10
        # Hold the closest points fixed in the target-attraction term of the equilibrium solve:
        # this seems to make the design optimization much more robust.
        uo.setHoldClosestPointsFixed(True, False)
        cb = pipeline_helper.UmbrellaOptimizationCallback(rest_height_optimizer, view, True, False, 1, rod_colors, doptays.record, tfview=tfview)
        algorithm = umbrella_optimization.OptAlgorithm.NEWTON_CG
        #algorithm = umbrella_optimization.OptAlgorithm.BFGS
        solverStatus = umbrella_optimization.optimize(rest_height_optimizer, algorithm, 1000, 0.005, 1e-5, cb, arm_length_lower_bound)
    
        f.write("Solver status for force: {} \n".format(solverStatus))

        force_pickle_path = '{}/{}_force_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

        force_rendering_path = '{}/{}_force_equilibrium_{}_rendering_output_{}.json.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

        save_data(curr_um, force_pickle_path, force_rendering_path, input_path, False, handleBoundary)

        import force_vector_visualization_helper
        importlib.reload(force_vector_visualization_helper)

        force_vector_visualization_helper.write_force_vector_visualization_file([tsf_pickle_path, force_pickle_path], ['{}/{}_tsf'.format(output_folder, name), '{}/{}_force'.format(output_folder, name)])

        f.write("Runnning this experiment on {} takes {} seconds\n".format(name, (datetime.now() - start_time).total_seconds()))

import sys

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) > 2:
        run_ablation(sys.argv[1], sys.argv[2] == "True")
    elif len(sys.argv) > 1:
        run_ablation(sys.argv[1], True)

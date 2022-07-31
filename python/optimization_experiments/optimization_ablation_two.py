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
import time

def visualize_ablation_result(file_name):
    import matplotlib.pyplot as plt
    import re
    # Prepare table
    columns = ('Objective Types', 'Elastic Energy', 'TSF', 'Forces')
    rows = ["TSF only", "TSF + Energy", "TSF + Energy + Forces", "TSF + Forces"]
    highlight = "#D9E76C"
    colors = [["w", "w", highlight ,"w"],
            ["w", highlight, highlight ,"w"],
            ["w", highlight, highlight ,highlight], 
            ["w", "w", highlight ,highlight]]

    cell_text = []
    with open(file_name, 'r') as f:
        content = f.readlines()
        row_counter = 0
        for line in content:
            if "Optimization objective" in line:
                values = re.split(": |, ", line)
                row_value = [rows[row_counter], np.round(float(values[4]), 6), np.round(float(values[8]), 6), np.round(float(values[10]), 6)]
                cell_text.append(row_value)
                row_counter += 1

    _, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text,cellColours=colors,
                        colLabels=columns,loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(4, 4)
    plt.show()

def run_optimization(curr_um_pickle_name, opt_obj_type, energy_weight, tsf_weight, force_weight, eqm_callback, opt_opts, fixedVars, output_folder, model_name, time_stamp, target_height_multiplier, algorithm, arm_length_lower_bound, input_path, handleBoundary, start_time, log_writer):
    curr_um = pickle.load(gzip.open(curr_um_pickle_name, 'r'))
    configure_umbrella_optimization(curr_um, bdryMultiplier = 1)

    results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = opt_opts, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

    opt_opts.niter = 50

    results.success

    optimizer = umbrella_optimization.UmbrellaOptimization(curr_um, opt_opts, 2.5, -1, False, fixedVars)
    rest_height_optimizer = umbrella_optimization.UmbrellaRestHeightsOptimization(optimizer)
    rest_height_optimizer.newPt(rest_height_optimizer.params())

    rest_height_optimizer.beta = tsf_weight
    rest_height_optimizer.gamma = energy_weight
    rest_height_optimizer.eta = 0
    rest_height_optimizer.zeta = 0# 1e1
    rest_height_optimizer.iota = force_weight

    import force_analysis
    cfm = force_analysis.UmbrellaForceMagnitudes(curr_um)

    normalActivationThreshold = min(np.percentile(cfm[:, 0], 30), 0)
    log_writer.write('normalActivationThreshold: {}\n'.format(normalActivationThreshold))

    rest_height_optimizer.objective.terms[-1].term.normalActivationThreshold = normalActivationThreshold

    rest_height_optimizer.objective.terms[-1].term.normalWeight = 1
    rest_height_optimizer.objective.terms[-1].term.tangentialWeight = 0
    rest_height_optimizer.objective.terms[-1].term.torqueWeight = 0


    doptays = DesignOptimizationAnalysis(rest_height_optimizer)

    pipeline_helper.prev_time_stamp = time.time()

    uo = rest_height_optimizer.get_parent_opt()

    uo.equilibriumOptimizer.options.verbose = 1
    #uo.equilibriumOptimizer.options.verboseWorkingSet = True
    uo.equilibriumOptimizer.options.gradTol = 1e-10
    # Hold the closest points fixed in the target-attraction term of the equilibrium solve:
    # this seems to make the design optimization much more robust.
    uo.setHoldClosestPointsFixed(True, False)
    cb = pipeline_helper.UmbrellaOptimizationCallback(rest_height_optimizer, None, True, False, 1, None, doptays.record, tfview=None)


    for weight in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        curr_um.attractionWeight = weight
        solverStatus = umbrella_optimization.optimize(rest_height_optimizer, algorithm, 300, 0.005, weight, cb, arm_length_lower_bound)
        print("Solver status for {}: {}\n".format(opt_obj_type, solverStatus))
        log_writer.write("Solver status for {}: {} \n".format(opt_obj_type, solverStatus))

    # Save each objective terms
    rest_height_optimizer.beta = 1e6
    rest_height_optimizer.gamma = 1
    rest_height_optimizer.eta = 0
    rest_height_optimizer.zeta = 0# 1e1
    rest_height_optimizer.iota = 1e10
    log_writer.write("Optimization objective with constant weight: {}\n".format(allDesignObjectives(rest_height_optimizer)))
    
    pickle_path = '{}/{}_{}_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder, model_name, opt_obj_type, time_stamp, target_height_multiplier)

    rendering_path = '{}/{}_{}_equilibrium_{}_rendering_output_{}.json.gz'.format(output_folder, model_name, opt_obj_type, time_stamp, target_height_multiplier)

    save_data(curr_um, pickle_path, rendering_path, input_path, False, handleBoundary)

    log_writer.write("Runnning this experiment on {} takes {} seconds\n".format(model_name, (datetime.now() - start_time).total_seconds()))


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

        # # ### Get true equilibrium state

        # curr_um.attractionWeight = 1e-7
        # with so(): results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

        input_pickle_path = '{}/{}_input_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

        input_rendering_path = '{}/{}_input_equilibrium_{}_rendering_output_{}.json.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

        save_data(curr_um, input_pickle_path, input_rendering_path, input_path, False, handleBoundary)
        
        ### Optimization
        algorithm = umbrella_optimization.OptAlgorithm.NEWTON_CG
        #algorithm = umbrella_optimization.OptAlgorithm.BFGS
        arm_length_lower_bound = input_data["plate_edge_length"] / 30 * 32

        import py_newton_optimizer
        opt_opts = py_newton_optimizer.NewtonOptimizerOptions()
        opt_opts.gradTol = 1e-8
        opt_opts.verbose = 10
        opt_opts.beta = 1e-6
        opt_opts.niter = 600
        opt_opts.verboseNonPosDef = False

        run_optimization(input_pickle_path, 'tsf_only', 0, 1e6, 0, eqm_callback, opt_opts, fixedVars, output_folder, name, time_stamp, target_height_multiplier, algorithm, arm_length_lower_bound, input_path, handleBoundary, datetime.now(), f)

        run_optimization(input_pickle_path, 'tsf+energy', 1, 1e6, 0, eqm_callback, opt_opts, fixedVars, output_folder, name, time_stamp, target_height_multiplier, algorithm, arm_length_lower_bound, input_path, handleBoundary, datetime.now(), f)

        run_optimization(input_pickle_path, 'tsf+energy+force', 1, 1e6, 1e8, eqm_callback, opt_opts, fixedVars, output_folder, name, time_stamp, target_height_multiplier, algorithm, arm_length_lower_bound, input_path, handleBoundary, datetime.now(), f)
        
        run_optimization(input_pickle_path, 'tsf+force', 0, 1e6, 1e8, eqm_callback, opt_opts, fixedVars, output_folder, name, time_stamp, target_height_multiplier, algorithm, arm_length_lower_bound, input_path, handleBoundary, datetime.now(), f)

        f.write("Runnning this experiment on {} takes {} seconds\n".format(name, (datetime.now() - start_time).total_seconds()))
        
import sys

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) > 2:
        run_ablation(sys.argv[1], sys.argv[2] == "True")
    elif len(sys.argv) > 1:
        run_ablation(sys.argv[1], True)

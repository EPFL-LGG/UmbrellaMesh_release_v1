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
def generate_force_vectors(name, handleBoundary, time_stamp):
    start_time = datetime.now()

    input_path = '../../data/{}.json.gz'.format(name)

    output_folder = '{}_{}'.format(time_stamp, name)

    io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path, handleBoundary=handleBoundary, handlePivots = True)

    tsf_pickle_path = '{}/{}_tsf+energy_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

    force_pickle_path = '{}/{}_tsf+energy+force_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder, name, time_stamp, target_height_multiplier)


    import force_vector_visualization_helper
    importlib.reload(force_vector_visualization_helper)

    force_vector_visualization_helper.write_force_vector_visualization_file([tsf_pickle_path, force_pickle_path], ['{}/{}_tsf'.format(output_folder, name), '{}/{}_force'.format(output_folder, name)])

def generate_input_true_equilibrium(name, handleBoundary, time_stamp):
    input_path = '../../data/{}.json.gz'.format(name)

    output_folder = '{}_{}'.format(time_stamp, name)

    io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path, handleBoundary=handleBoundary, handlePivots = True)
    input_pickle_path = '{}/{}_input_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(output_folder, name, time_stamp, target_height_multiplier)
    input_rendering_path = '{}/{}_input_equilibrium_{}_rendering_output_{}.json.gz'.format(output_folder, name, time_stamp, target_height_multiplier)

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
    # ### Get true equilibrium state
    curr_um.attractionWeight = 1e-9
    with so(): results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)
    print(results.success)
    save_data(curr_um, input_pickle_path, input_rendering_path, input_path, False, handleBoundary)


# generate_force_vectors("hemisphere_5t", False, "2022_05_02_00_38")
# generate_input_true_equilibrium("hemisphere_5t", False, "2022_05_02_00_38")

# generate_force_vectors("lilium_smooth", False, "2022_05_02_11_08")
# generate_input_true_equilibrium("lilium_smooth", False, "2022_05_02_11_08")

generate_force_vectors("hive", True, "2022_05_02_11_35")
generate_input_true_equilibrium("hive", True, "2022_05_02_11_35")

# generate_force_vectors("lilium_smooth", False, "2022_05_02_00_38")
# generate_force_vectors("hive_10t", False, "2022_05_02_00_38")
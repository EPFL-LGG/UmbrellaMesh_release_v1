import umbrella_optimization, umbrella_mesh
from umbrella_optimization import OptEnergyType

import numpy as np
import numpy.linalg as la
import time
import os

prev_vars = None
prev_time_stamp = None

def allEnergies(um):
    return {name: um.energy(uet) for name, uet in umbrella_mesh.UmbrellaEnergyType.__members__.items()}

def allGradientNorms(um):
    return {name: la.norm(um.gradient(umbrellaEnergyType = uet)) for name, uet in umbrella_mesh.UmbrellaEnergyType.__members__.items()}

def allDesignObjectives(opt):
    v = opt.objective.weightedValues()
    v['Full'] = sum(v.values())
    return v

def allDesignGradientNorms(opt):
    return {name: la.norm(opt.gradp_J(opt.params(), uet)) for name, uet in umbrella_optimization.OptEnergyType.__members__.items()}

import tri_mesh_viewer
class TargetFittingVisualization():
    def __init__(self, um, tsf, superview):
        self.um, self.tsf = um, tsf
        self.view = tri_mesh_viewer.LineMeshViewer(self.getLines(), superView=superview)
    def update(self):
        self.view.mesh.vertices = self.getLines()[0]
        self.view.update()
    def getLineV(self): return np.array(np.vstack([self.tsf.getQueryPtPos(self.um).reshape(-1, 3),
                                                   self.tsf.query_pt_pos_tgt.reshape(-1, 3),
                                                   self.tsf.umbrella_closest_surf_pts.reshape(-1,3)]), dtype=np.float32)
    def getLines(self):
        numPts = self.tsf.numQueryPt(self.um)
        lineE = np.vstack([np.column_stack([np.arange(numPts), np.arange(numPts) +     numPts]),
                           np.column_stack([np.arange(numPts), np.arange(numPts) + 2 * numPts])])
        return self.getLineV(), lineE

class UmbrellaOptimizationCallback:
    def __init__(self, optimizer, view, update_color = False, no_surface = False, callback_freq = 1, rod_colors = None, recorder = None, tfview = None, osrender = None):
        self.optimizer     = optimizer
        self.umbrella      = optimizer.committedObject
        self.view          = view
        self.tfview        = tfview
        self.update_color  = update_color
        self.no_surface    = no_surface
        self.callback_freq = callback_freq
        self.iterateData   = []
        self.rod_colors    = rod_colors
        self.recorder      = recorder
        self.osrender      = osrender

    def __call__(self):
        global prev_vars
        global prev_time_stamp
        if self.view and (len(self.iterateData) % self.callback_freq == 0):
            if self.tfview is not None: self.tfview.update()
            if self.update_color and self.rod_colors is not None:
                self.view.showScalarField(self.rod_colors)
            else:
                pass
                self.view.update()

        curr_vars = self.umbrella.getExtendedDoFsPARL()
        # Record values of all objective terms, plus timestamp and variables.
        idata = allDesignObjectives(self.optimizer)
        idata.update({'iteration_time':   time.time() - prev_time_stamp,
                      'extendedDoFsPARL': curr_vars, 
                      'designParams': self.optimizer.params()}) 
        idata.update({f'{name}_grad_norm': gn for name, gn in allDesignGradientNorms(self.optimizer).items()})
        self.iterateData.append(idata)
        prev_time_stamp = time.time()
        if self.recorder is not None:
            self.recorder()
        if self.osrender is not None:
            self.osrender()
        return

    def numIterations(self): return len(self.iterateData)

def get_objective_components(iterateData, vs):
    opt_objective_elastic = []
    opt_objective_target = []
    opt_objective_deployment = []

    for iter_idx in range(len(iterateData)):
        opt_objective_elastic.append(iterateData[iter_idx]['ElasticEnergy'])
        opt_objective_target.append(iterateData[iter_idx]['TargetFitting'])
        opt_objective_deployment.append(iterateData[iter_idx]['DeploymentForce'])
        
    opt_objective_elastic = np.array(opt_objective_elastic)
    opt_objective_target = np.array(opt_objective_target)
    opt_objective_deployment = np.array(opt_objective_deployment)

    opt_total_objective =  np.array([opt_objective_elastic, opt_objective_target, opt_objective_deployment]).sum(axis=0)
    
    colors = [vs.elastic_color, vs.target_color, vs.deployment_color]
    labels = [vs.elastic_label, vs.target_label, vs.deployment_label]
    return opt_objective_elastic, opt_objective_target, opt_objective_deployment, opt_total_objective, colors, labels

def get_grad_norm_components(iterateData, vs):
    opt_grad_norm_elastic = []
    opt_grad_norm_target = []
    opt_grad_norm_deployment = []

    for iter_idx in range(len(iterateData)):
        opt_grad_norm_elastic.append(iterateData[iter_idx]['ElasticEnergy_grad_norm'])
        opt_grad_norm_target.append(iterateData[iter_idx]['TargetFitting_grad_norm'])
        opt_grad_norm_deployment.append(iterateData[iter_idx]['DeploymentForce_grad_norm'])
        
    opt_grad_norm_elastic = np.array(opt_grad_norm_elastic)
    opt_grad_norm_target = np.array(opt_grad_norm_target)
    opt_grad_norm_deployment = np.array(opt_grad_norm_deployment)

    opt_total_grad_norm =  np.array([opt_grad_norm_elastic, opt_grad_norm_target, opt_grad_norm_deployment]).sum(axis=0)
    
    colors = [vs.elastic_color, vs.target_color, vs.deployment_color]
    labels = [vs.elastic_label, vs.target_label, vs.deployment_label]
    return opt_grad_norm_elastic, opt_grad_norm_target, opt_grad_norm_deployment, opt_total_grad_norm, colors, labels

import matplotlib.pyplot as plt

class Visualization_Setting():
    def __init__(self):
        self.cmap = plt.get_cmap("Set2")
        self.elastic_color = '#555358'
        self.target_color = self.cmap(1)
        self.deployment_color = self.cmap(2)
       
        self.elastic_label = 'Elastic Energy'
        self.target_label = 'Target Surface Fitting'
        self.deployment_label = 'Deployment Force'
        
        self.x_label = 'Iteration'
        self.figure_size = (17, 6)
        self.figure_label_size = 30

def plot_objective(vs, total_objective, figure_name, label, grad_norm = False):
    if len(total_objective) == 0:
        print("No data available!")
        return
    fig, host = plt.subplots()
    cmap = plt.get_cmap("Set2")
    x=range(len(total_objective))
    y=np.array(total_objective)

    plt.plot(x,y)
    fig.set_size_inches(vs.figure_size)
    plt.ylabel('Grad Norm' if grad_norm else 'Objective Value', fontsize = vs.figure_label_size)
    plt.title(label, fontsize = vs.figure_label_size)
    fig.set_size_inches(vs.figure_size)
    fig.savefig(figure_name, dpi=200)
    plt.close()

def plot_objective_stack(vs, total_objective, objective_components_list, color_list, label_list, figure_name, label, grad_norm = False, iteration = None):
    fig, host = plt.subplots()
    fig.set_size_inches(vs.figure_size)

    x=range(len(total_objective))
    y=np.array(objective_components_list)
     
    # Basic stacked area chart.
    plt.stackplot(x,y, labels=label_list, colors = color_list, baseline='zero')
    if iteration != None:
        host.axvline(iteration, alpha=1, color=vs.elastic_color)
    plt.legend(loc='upper right', prop={'size': 15}, fancybox=True)
    plt.ylabel('Grad Norm' if grad_norm else 'Objective Value', fontsize = vs.figure_label_size)
    plt.title(label, fontsize = vs.figure_label_size)
    fig.savefig(figure_name, bbox_inches='tight', dpi=200)
    plt.close()

def show_center_joint_normal(umbrella, joint_list):
    joint_vector_field = [np.zeros((s.rod.numVertices(), 3)) for s in umbrella.segments()]
    for ji in range(umbrella.numJoints()):
        if ji in joint_list:
            if (umbrella.joint(ji).valence() != 3):
                continue
            seg_index = umbrella.joint(ji).getSegmentAt(0)
            if umbrella.segment(seg_index).startJoint == ji:
                vx_index = 0
            else:
                vx_index = -1
            joint_vector_field[seg_index][vx_index] = umbrella.joint(ji).ghost_normal()
    return joint_vector_field

def set_joint_vector_field(umbrella, umbrella_view, joint_vector_field):
    vector_field = [np.zeros((s.rod.numVertices(), 3)) for s in umbrella.segments()]
    for ji in range(umbrella.numJoints()):
        seg_index = umbrella.joint(ji).getSegmentAt(0)
        if umbrella.segment(seg_index).startJoint == ji:
            vx_index = 0
        else:
            vx_index = -1
        vector_field[seg_index][vx_index] = joint_vector_field[ji]
    
    umbrella_view.update(vectorField = vector_field)

def show_joint_normal(umbrella, joint_list):
    joint_vector_field = [np.zeros((s.rod.numVertices(), 3)) for s in umbrella.segments()]
    for ji in range(umbrella.numJoints()):
        if ji in joint_list:
            seg_index = umbrella.joint(ji).getSegmentAt(0)
            if umbrella.segment(seg_index).startJoint == ji:
                vx_index = 0
            else:
                vx_index = -1
            joint_vector_field[seg_index][vx_index] = umbrella.joint(ji).ghost_normal()
    return joint_vector_field



import pickle 
import gzip

def save_data(umbrella, output_pickle_path, output_rendering_path, input_json_path, is_rest_state, handleBoundary):
    pickle.dump(umbrella, gzip.open(output_pickle_path, 'w'))

    from load_jsondata import write_deformed_config

    write_deformed_config(umbrella, input_json_path, output_rendering_path, write_stress = False, is_rest_state = is_rest_state, handleBoundary = handleBoundary)

    

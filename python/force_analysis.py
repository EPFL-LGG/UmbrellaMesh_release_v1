import numpy as np
import numpy.linalg as la
import umbrella_mesh

def get_gradient_per_force_type(linkage, force_type):
    LEA = linkage.getLinearActuator()
    save_angleStiffness = LEA.angleStiffness
    save_axialStiffness = LEA.axialStiffness
    save_tangentialStiffness = LEA.tangentialStiffness

    LEA.angleStiffness = 0
    LEA.axialStiffness = 0
    LEA.tangentialStiffness = 0
    
    if force_type == 'angle':
        LEA.angleStiffness = save_angleStiffness
    if force_type == 'axial':
        LEA.axialStiffness = save_axialStiffness
    if force_type == 'tangential':
        LEA.tangentialStiffness = save_tangentialStiffness
    
    g = linkage.linearActuatorGradient()

    LEA.angleStiffness = save_angleStiffness
    LEA.axialStiffness = save_axialStiffness
    LEA.tangentialStiffness = save_tangentialStiffness
    return g


def deploymentForceFields(linkage):
    g = get_gradient_per_force_type(linkage, 'axial')
    axialForce = []
    for ui in range(linkage.numUmbrellas()):
        top_ji = linkage.getUmbrellaCenterJi(ui, 0)
        axialForce.append(g[linkage.dofOffsetForJoint(top_ji): linkage.dofOffsetForJoint(top_ji) + 3])
        
    g = get_gradient_per_force_type(linkage, 'angle')
    angleForce = []
    for ui in range(linkage.numUmbrellas()):
        top_ji = linkage.getUmbrellaCenterJi(ui, 0)
        angleForce.append(g[linkage.dofOffsetForJoint(top_ji) + 3: linkage.dofOffsetForJoint(top_ji) + 6])
        
    g = get_gradient_per_force_type(linkage, 'tangential')
    tangentialForce = []
    for ui in range(linkage.numUmbrellas()):
        top_ji = linkage.getUmbrellaCenterJi(ui, 0)
        tangentialForce.append(g[linkage.dofOffsetForJoint(top_ji): linkage.dofOffsetForJoint(top_ji) + 3])

    nj = linkage.numJoints()
    axialForceField = []
    tangentialForceField = []
    torqueField = []
    for si, s in enumerate(linkage.segments()):
        ne = s.rod.numEdges()
        af = np.zeros((ne, 3))
        tf = np.zeros((ne, 3))
        torque = np.zeros((ne, 3))
        for endpt, ji in enumerate([s.startJoint, s.endJoint]):
            if (ji > nj): continue
            j = linkage.joint(ji)
            if (j.jointType() == umbrella_mesh.JointType.Rigid and j.jointPosType() == umbrella_mesh.JointPosType.Top):
                terminalEdge = ne - 1 if endpt else 0
                af[terminalEdge] = axialForce[j.umbrellaID()[0]]
                tf[terminalEdge] = tangentialForce[j.umbrellaID()[0]]
                torque[terminalEdge] = angleForce[j.umbrellaID()[0]]

        axialForceField.append(af)
        tangentialForceField.append(tf)
        torqueField.append(torque)
    return {'axial': axialForceField,
            'tangential': tangentialForceField, 
            'torque': torqueField}

import linkage_vis
from ipywidgets import HBox
class deploymentForceFieldVisualization():
    def __init__(self, linkage):
        self.forces = deploymentForceFields(linkage)
        self.axialView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['axial'])
        self.tangentialView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['tangential'])
        self.torqueView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['torque'])
        self.axialView.averagedMaterialFrames = True
        self.tangentialView.averagedMaterialFrames = True
        self.torqueView.averagedMaterialFrames = True

    def maxForce(self):
        return (np.max([np.linalg.norm(sf, axis=1) for sf in self.forces['axial']]),
                np.max([np.linalg.norm(tf, axis=1) for tf in self.forces['tangential']]),
                np.max([np.linalg.norm(tf, axis=1) for tf in self.forces['torque']]))

    def show(self):
        return HBox([self.axialView.show(), self.tangentialView.show(), self.torqueView.show()])


def deploymentForceMagnitudes(linkage):
    g = get_gradient_per_force_type(linkage, 'axial')
    axialForce = []
    for ui in range(linkage.numUmbrellas()):
        top_ji = linkage.getUmbrellaCenterJi(ui, 0)
        axialForce.append(g[linkage.dofOffsetForJoint(top_ji): linkage.dofOffsetForJoint(top_ji) + 3])
        
    g = get_gradient_per_force_type(linkage, 'angle')
    angleForce = []
    for ui in range(linkage.numUmbrellas()):
        top_ji = linkage.getUmbrellaCenterJi(ui, 0)
        angleForce.append(g[linkage.dofOffsetForJoint(top_ji) + 3: linkage.dofOffsetForJoint(top_ji) + 6])
        
    g = get_gradient_per_force_type(linkage, 'tangential')
    tangentialForce = []
    for ui in range(linkage.numUmbrellas()):
        top_ji = linkage.getUmbrellaCenterJi(ui, 0)
        tangentialForce.append(g[linkage.dofOffsetForJoint(top_ji): linkage.dofOffsetForJoint(top_ji) + 3])

    result = []
    for ui in range(linkage.numUmbrellas()):
        top_ji = linkage.getUmbrellaCenterJi(ui, 0)
        af = axialForce[ui]
        tf = tangentialForce[ui]
        torque = angleForce[ui]
        separationDirection = linkage.joint(top_ji).ghost_normal() 
        separationForce = -af.dot(separationDirection)
        result.append([separationForce, la.norm(tf), la.norm(torque)])
    return np.array(result)

from matplotlib import pyplot as plt
def deploymentForceAnalysis(linkage):
    cfm = deploymentForceMagnitudes(linkage)
    separationForce = cfm[:, 0]
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.xlim((separationForce.min(), max([0, separationForce.max()])))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.title('Separation Forces')
    plt.xlabel('Separation Force Mag.')
    plt.ylabel('Number of Umbrellas')
    plt.hist(separationForce, 200);

    plt.subplot(1, 3, 2)
    plt.title('Tangential Forces')
    plt.xlabel('Tangential Force Mag.')
    plt.ylabel('Number of Umbrellas')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.hist(cfm[:, 1], 100);
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.title('Torques')
    plt.xlabel('Torque Mag.')
    plt.ylabel('Number of Umbrellas')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.hist(cfm[:, 2], 100);
    plt.tight_layout()


# Umbrella Forces

def UmbrellaForceFields(linkage):
    AForceOnJoint = linkage.UmbrellaRivetNetForceAndTorques()[:, 0:3]
    ATorqueOnJoint = linkage.UmbrellaRivetNetForceAndTorques()[:, 3:6]

    nj = linkage.numJoints()
    separationForceField = []
    tangentialForceField = []
    torqueField = []
    for si, s in enumerate(linkage.segments()):
        ne = s.rod.numEdges()
        sf = np.zeros((ne, 3))
        tf = np.zeros((ne, 3))
        torque = np.zeros((ne, 3))
        # for endpt, ji in enumerate([s.startJoint, s.endJoint]):
        #     if (ji > nj): continue
        #     j = linkage.joint(ji)
        #     if (j.jointType() == umbrella_mesh.JointType.Rigid and (j.jointPosType() == umbrella_mesh.JointPosType.Top or j.jointPosType() == umbrella_mesh.JointPosType.Bot) and j.valence() == 3):
        #         forceIndex = 0 if j.jointPosType() == umbrella_mesh.JointPosType.Top else 1
        #         terminalEdge = ne - 1 if endpt else 0
        #         f = AForceOnJoint[j.umbrellaID()[0] * 2 + forceIndex]
        #         n = j.ghost_normal()
        #         t = ATorqueOnJoint[j.umbrellaID()[0] * 2 + forceIndex]
        #         sf[terminalEdge] = np.clip(f.dot(n), 0, None) * n
        #         tf[terminalEdge] = f - f.dot(n) * n
        #         torque[terminalEdge] = t

        separationForceField.append(sf)
        tangentialForceField.append(tf)
        torqueField.append(torque)

    for udi in range(len(AForceOnJoint)):
        f = AForceOnJoint[udi]
        torque = ATorqueOnJoint[udi]
        ji = linkage.getUmbrellaCenterJi(int(udi / 2), udi % 2)  
        j = linkage.joint(ji)
        n = j.ghost_normal()

        si = j.getSegmentAt(0)
        s = linkage.segment(si)
        ne = s.rod.numEdges()
        terminalEdge = 0 if (linkage.segment(si).startJoint == ji) else ne - 1
        separationForceField[si][terminalEdge] = np.clip(f.dot(n), 0, None) * n
        tangentialForceField[si][terminalEdge] = f - f.dot(n) * n
        torqueField[si][terminalEdge] = torque
            
    return {'separation': separationForceField,
            'tangential': tangentialForceField, 
            'torque': torqueField}

class UmbrellaForceFieldVisualization():
    def __init__(self, linkage):
        self.forces = UmbrellaForceFields(linkage)
        self.separationView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['separation'])
        self.tangentialView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['tangential'])
        self.torqueView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['torque'])
        self.separationView.averagedMaterialFrames = True
        self.tangentialView.averagedMaterialFrames = True
        self.torqueView.averagedMaterialFrames = True

    def maxForce(self):
        return (np.max([np.linalg.norm(sf, axis=1) for sf in self.forces['separation']]),
                np.max([np.linalg.norm(tf, axis=1) for tf in self.forces['tangential']]),
                np.max([np.linalg.norm(tf, axis=1) for tf in self.forces['torque']]))

    def show(self):
        return HBox([self.separationView.show(), self.tangentialView.show(), self.torqueView.show()])

def UmbrellaForceMagnitudes(linkage):
    AForceOnJoint = linkage.UmbrellaRivetNetForceAndTorques()[:, 0:3]
    ATorqueOnJoint = linkage.UmbrellaRivetNetForceAndTorques()[:, 3:6]

    result = []
    for udi in range(len(AForceOnJoint)):
        f = AForceOnJoint[udi]
        torque = ATorqueOnJoint[udi]
        ji = linkage.getUmbrellaCenterJi(int(udi / 2), udi % 2)  
        j = linkage.joint(ji)
        separationDirection = j.ghost_normal()
        separationForce = f.dot(separationDirection)
        tangentialForce = np.sqrt(f.dot(f) - separationForce**2)
        result.append([separationForce, tangentialForce, la.norm(torque)])
    return np.array(result)

def UmbrellaForceAnalysis(linkage):
    cfm = UmbrellaForceMagnitudes(linkage)
    separationForce = cfm[:, 0]
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.xlim((separationForce.min(), max([0, separationForce.max()])))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.title('Separation Forces')
    plt.xlabel('Separation Force Mag.')
    plt.ylabel('Number of Umbrellas')
    plt.hist(separationForce, 200);

    plt.subplot(1, 3, 2)
    plt.title('Tangential Forces')
    plt.xlabel('Tangential Force Mag.')
    plt.ylabel('Number of Umbrellas')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.hist(cfm[:, 1], 100);
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.title('Torques')
    plt.xlabel('Torque Mag.')
    plt.ylabel('Number of Umbrellas')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.hist(cfm[:, 2], 100);
    plt.tight_layout()



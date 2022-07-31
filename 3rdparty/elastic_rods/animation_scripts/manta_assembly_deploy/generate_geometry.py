import sys, os; sys.path.append('../python')
import numpy as np, elastic_rods
from bending_validation import suppress_stdout
from linkage_vis import LinkageViewer

l = elastic_rods.RodLinkage('../examples/florin/20181227_193550_meshID_5ca2f7ab-3602-4ede-ac4d-c2bd798b2961.obj', 8)
driver=l.centralJoint()

mat = elastic_rods.RodMaterial('rectangle', 20000, 0.3, [10, 7])
l.setMaterial(mat)

with suppress_stdout(): elastic_rods.restlen_solve(l)
jdo = l.dofOffsetForJoint(driver)
fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion for a single joint
fixedVars.append(jdo + 6) # constrain angle at the driving joint
with suppress_stdout(): elastic_rods.compute_equilibrium(l, fixedVars=fixedVars)

from open_linkage import open_linkage
def equilibriumSolver(tgtAngle, l, opts, fv):
    opts.beta = 1e-8
    opts.gradTol = 1e-4
    opts.useIdentityMetric = False
    return elastic_rods.compute_equilibrium(l, tgtAngle, options=opts, fixedVars=fv)

# Open the manta ray a little so that it doesn't self-collide too much 
with suppress_stdout(): open_linkage(l, driver, np.deg2rad(22) -
        l.averageJointAngle, 10, zPerturbationEpsilon=0,
        equilibriumSolver=equilibriumSolver,
        maxNewtonIterationsIntermediate=20, verbose=10,
        useTargetAngleConstraint=True);

from glob import glob
def prepareDirectory(name):
    if (os.path.exists(name)):
        for f in glob(name + '/*.msh'):
            os.remove(f)
    else: os.mkdir(name)

def countFiles(dirname):
    return len(glob(dirname + '/*.msh'))

def padAnimation(name, requestedFrames, zeroIndex = True):
    numFrames = countFiles(name)
    paddingFrames = requestedFrames - numFrames

    for f in range(paddingFrames):
        padFileName = '{}/frame_{}.msh'.format(name, f + numFrames + (0 if zeroIndex else 1))
        print("Added padding frame {}".format(padFileName))
        l.saveVisualizationGeometry(padFileName, averagedMaterialFrames=True)

prepareDirectory('manta_ray_assembly_drop')

import drop_animation
drop_animation.drop_animation('manta_ray_assembly_drop', l, phaseOffset=0.17, flipZ=True)
# Pad animation to 5s
padAnimation('manta_ray_assembly_drop', int(4.5 * 30))

prepareDirectory('manta_ray_deploy')
with suppress_stdout(): open_linkage(l, driver, np.deg2rad(93) - l.averageJointAngle, int(2.5 * 30), zPerturbationEpsilon=0, equilibriumSolver=equilibriumSolver, maxNewtonIterationsIntermediate=200, verbose=10, useTargetAngleConstraint=True, outPathFormat='manta_ray_deploy/frame_{}.msh')
# Pad animation to 5s
padAnimation('manta_ray_deploy', int(5.5 * 30), False)

import shutil, os
from glob import glob
import numpy as np

def countFiles(dirname):
    return len(glob(dirname + '/*.msh'))

if os.path.exists('frames'):
    shutil.rmtree('frames')

os.mkdir('frames')

maxNum = 0
for f in glob('manta_ray_assembly_drop/*.msh'):
    num = int(f.split('_')[-1].split('.')[0])
    maxNum = max(maxNum, num)
    shutil.copy('manta_ray_assembly_drop/frame_{}.msh'.format(num), 'frames/frame_{}.msh'.format(num))

# Note: this assumes manta_ray_deploy files are 1-indexed
for f in glob('manta_ray_deploy/*.msh'):
    num = int(f.split('_')[-1].split('.')[0])
    shutil.copy('manta_ray_deploy/frame_{}.msh'.format(num), 'frames/frame_{}.msh'.format(maxNum + num))

# Generate a gmsh file for each frame with the proper rotation
import string
template = string.Template(''.join(open('manta_render_template.opt', 'r').readlines()))

numFrames = len(glob('frames/*.msh'))

# Quaternions representing the camera orientation
import scipy
from scipy.spatial.transform import Rotation as R, Slerp
q1 = [-0.4936907408713928, 0.5857408574440296, -0.4936907408713928, 0.411638861960874]
q2 = [-0.1998909798361335, 0.7395042230315945, -0.6214018631464604, 0.1644287832974986]

interp = Slerp([0, 1], R.from_quat(np.array([q1, q2])))
interp_quats = interp(np.linspace(0, 1, numFrames)).as_quat()

for f in range(numFrames):
    q = list(map(str, interp_quats[f]))
    open('frames/render_{}.opt'.format(f), 'w').write(template.substitute({'Q0': q[0], 'Q1': q[1], 'Q2': q[2], 'Q3': q[3]}))

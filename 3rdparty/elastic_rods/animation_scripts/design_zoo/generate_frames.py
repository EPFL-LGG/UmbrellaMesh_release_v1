import shutil, os, sys
from glob import glob
import numpy as np

def countFiles(dirname):
    return len(glob(dirname + '/*.msh'))

if (len(sys.argv) != 3):
    raise Exception('usage: generate_frames.py ExampleName numFrames')

name, numFrames = sys.argv[1], int(sys.argv[2])

OrientationKeyframes = {
    'ElephantEars': [[0.1576451309038442, -0.4653093177897976, -0.827972913219713, 0.2703629161974069], [0.01357614834443358, -0.6329918945855493, -0.7737298442968024, 0.02188784204920054]],
    '2Layer2Bump': [[0.8862872085350852, 0.1456944296878862, -0.1543764693870573, -0.4116260716288422], [0.6728472446083389, 0.3697739694827623, -0.4259087115133672, -0.4786915148328386]],
    'AsymmetricClapper2': [[-0.2786063407018196, -0.7757339370555509, 0.4949527279296849, 0.2750221135399839], [-0.08879284703958949, -0.7785818669733213, 0.5935305854116829, 0.1834326876971704]],
    'ScaryFace': [[-0.2612684325547747, 0.09367646519484278, 0.957278396332169, -0.08112704810778382], [0, 0.1736481776669302, 0.9848077530122081, 0]],
    'NegCRVwithBreatherHole': [[0.6840784368776633, 0.4625016577305032, -0.2267047472176375, -0.5164628412427532], [0.5538529198476108, 0.5777074843599438, -0.3743060480938284, -0.4683972545299129]],
    'swimming_ray': [[0.3157388171045458, 0.7820601832619865, -0.4982272849178697, -0.2011478105598424], [0, 0.766044443118978, -0.6427876096865394, 0]]
}

if (name not in OrientationKeyframes):
    raise Exception('Uknown example name {}'.format(name))


directory = 'frames/{}'.format(name)
if os.path.exists(directory):
    shutil.rmtree(directory)

if not os.path.exists('frames'): os.mkdir('frames')
os.mkdir(directory)

# Quaternions representing the camera orientation
import scipy
from scipy.spatial.transform import Rotation as R, Slerp

# Generate a gmsh file for each frame with the proper rotation
import string
template = string.Template(''.join(open('render_templates/{}.opt'.format(name), 'r').readlines()))

q1, q2 = OrientationKeyframes[name]

interp = Slerp([0, 1], R.from_quat(np.array([q1, q2])))
interp_quats = interp(np.linspace(0, 1, numFrames)).as_quat()

opt_paths = []
png_paths = []
for f in range(numFrames):
    q = list(map(str, interp_quats[f]))
    path = 'frames/{}/render_{}.opt'.format(name, f)
    opt_paths.append(path)
    png_paths.append('frames/{}/{}.png'.format(name, f))
    open(path, 'w').write(template.substitute({'Q0': q[0], 'Q1': q[1], 'Q2': q[2], 'Q3': q[3]}))

import subprocess, multiprocessing

def renderFrame(args):
    opts, pngPath = args
    subprocess.call(['gmsh_offscreen', '-n', 'meshes/{}.msh'.format(name), opts, '-o', pngPath])
    subprocess.call(['mogrify', '-resize', '50%', pngPath])

with multiprocessing.Pool(processes=4) as pool:
    pool.map(renderFrame, zip(opt_paths, png_paths))

subprocess.call(['ffmpeg', '-f', 'image2',
                 '-framerate', '30',
                 '-i', 'frames/{}/%d.png'.format(name),
                 '-c:v', 'libx264',
                 '-preset', 'veryslow',
                 '-qp', '18',
                 '-pix_fmt', 'yuv420p',
                 '{}.mp4'.format(name)])

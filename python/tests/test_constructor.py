import sys
sys.path.append('..')
import umbrella_mesh

from load_jsondata import read_data
def construct_umbrella(input_path):
	input_data, io = read_data(filepath = input_path)
	curr_um = umbrella_mesh.UmbrellaMesh(io)
	thickness = io.material_params[6]
	curr_um.targetDeploymentHeight = thickness * 1
	return curr_um

def test_umbrella():
	construct_umbrella('../../data/hemisphere_5t.json.gz')

if __name__ == "__main__":
	test_umbrella()
import numpy as np
import numpy.linalg as la
import copy
import matplotlib.pyplot as plt

def normalize(vec):
        return vec / la.norm(vec)

def fill_plate_vars_with_center(umesh, joint_field):
    scalar_field = [np.zeros(s.rod.numVertices()) for s in umesh.segments()]
    for ji in range(umesh.numJoints()):
        joint = umesh.joint(ji)
        if joint.valence() == 3:
            for local_seg_id in range(3):
                seg_id = joint.getSegmentAt(local_seg_id)
                scalar_field[seg_id][:] = joint_field[ji]
    return scalar_field

def get_seg_normals(umesh, seg_id):
    curr_seg = umesh.segment(seg_id)
    curr_rod = curr_seg.rod
    dc = curr_rod.deformedConfiguration()
    assert curr_seg.startJoint < umesh.numJoints() and curr_seg.endJoint < umesh.numJoints()
    curr_normals = []
    curr_normals.append(dc.materialFrame[0].d2)
    for edge_index in range(len(dc.materialFrame))[1:]:
        curr_normals.append(normalize(dc.materialFrame[edge_index-1].d2 + dc.materialFrame[edge_index].d2))
    curr_normals.append(dc.materialFrame[-1].d2)

    return np.array(curr_normals)
def compute_plate_normals(umesh):
    plate_normals = []
    for seg_id in range(umesh.numSegments()):
        plate_normals.append(np.zeros((umesh.segment(seg_id).rod.numVertices(), 3)))

    for jid in range(umesh.numJoints()):
        joint = umesh.joint(jid)
        if joint.valence() == 3:
            for local_seg_id in range(3):
                seg_id = joint.getSegmentAt(local_seg_id)
                segment = umesh.segment(seg_id)
                plate_normals[seg_id] = get_seg_normals(umesh, seg_id)
    return plate_normals

def compute_avg_plate_normals(umesh, plate_normals = None):
    if plate_normals is None:
        plate_normals = compute_plate_normals(umesh)
    avg_plate_normal = np.zeros((umesh.numJoints(), 3))
    # Compute mean normal at plate center
    for jid in range(umesh.numJoints()):
        joint = umesh.joint(jid)
        if joint.valence() == 3:
            for local_seg_id in range(3):
                seg_id = joint.getSegmentAt(local_seg_id)
                curr_normals = copy.copy(plate_normals[seg_id])
                if local_seg_id > 0:
                    if joint.getIsStartAt(local_seg_id):
                        avg_plate_normal[jid] += curr_normals[1:].sum(axis = 0)
                    else:
                        avg_plate_normal[jid] += curr_normals[:-1].sum(axis = 0)
                else:
                    avg_plate_normal[jid] += curr_normals.sum(axis = 0)
            avg_plate_normal[jid] = normalize(avg_plate_normal[jid])
    return avg_plate_normal
def compute_plate_normal_variation(umesh):
    plate_normals = compute_plate_normals(umesh)
    avg_plate_normal = compute_avg_plate_normals(umesh, plate_normals = plate_normals)
    # Compute average and max angle distortion
    avg_plate_angle = np.zeros((umesh.numJoints()))
    max_plate_angle = np.zeros((umesh.numJoints()))
    normal_angle = np.arccos(avg_plate_normal.dot(np.array([0,0,1.0])))
    
    for jid in range(umesh.numJoints()):
        joint = umesh.joint(jid)
        if joint.valence() == 3:
            num_angles = 0
            for local_seg_id in range(3):
                seg_id = joint.getSegmentAt(local_seg_id)
                curr_normals = plate_normals[seg_id]
                angles = np.arccos(np.dot(curr_normals, avg_plate_normal[jid]))

                if joint.getIsStartAt(local_seg_id):
                    avg_plate_angle[jid] += angles[1:].sum()
                    max_plate_angle[jid] = max(max_plate_angle[jid], angles[1:].max())
                else:
                    avg_plate_angle[jid] += angles[:-1].sum()
                    max_plate_angle[jid] = max(max_plate_angle[jid], angles[:-1].max())
                num_angles += len(angles) - 1
            avg_plate_angle[jid] /= num_angles
    return normal_angle, avg_plate_angle, max_plate_angle
    return fill_plate_vars_with_center(umesh, normal_angle), fill_plate_vars_with_center(umesh, avg_plate_angle), fill_plate_vars_with_center(umesh, max_plate_angle)

def get_plate_center_joint_mask(umesh):
    mask = np.zeros(umesh.numJoints()).astype(np.bool)
    for jid in range(umesh.numJoints()):
        joint = umesh.joint(jid)
        if joint.valence() == 3:
            mask[jid] = 1
    return mask

def plate_top_bot_angle(umesh, top_bot_map):
    avg_normals = compute_avg_plate_normals(umesh)
    top_bot_angle = np.zeros((umesh.numJoints()))
    normal_pos_angle = np.zeros((umesh.numJoints()))
    top_bot_proj_center_offset = np.zeros((umesh.numJoints()))
    for [top_id, bot_id] in top_bot_map:
        top_normal, bot_normal = avg_normals[top_id], avg_normals[bot_id]
        top_pos, bot_pos = umesh.joint(top_id).position, umesh.joint(bot_id).position
        top_bot_vec = normalize(top_pos - bot_pos)
        
        top_bot_angle[top_id] = np.arccos(np.dot(top_normal, -bot_normal))
        top_bot_angle[bot_id] = top_bot_angle[top_id]
        
        normal_pos_angle[top_id] = np.arccos(np.dot(top_normal, top_bot_vec))
        normal_pos_angle[bot_id] = np.arccos(np.dot(-bot_normal, top_bot_vec))
        
        top_bot_proj_center_offset[top_id] = np.linalg.norm(top_pos - bot_pos)*np.sin(normal_pos_angle[top_id])
        top_bot_proj_center_offset[bot_id] = np.linalg.norm(top_pos - bot_pos)*np.sin(normal_pos_angle[bot_id])
        
    return top_bot_angle, normal_pos_angle, top_bot_proj_center_offset # alpha, beta, e
        
def plot_alpha_beta_e(umesh, alpha, beta, e, thickness, deployment_ratio):
    mask = get_plate_center_joint_mask(umesh)
    fig, axs = plt.subplots(1, 3, figsize = (30, 10))
    axs[0].hist(alpha[mask]*180/np.pi, label = 'alpha')
    axs[0].set_title(r'$\alpha$ (in degrees)', fontsize = 20.0)
    axs[1].hist(beta[mask]*180/np.pi, label = 'beta')
    axs[1].set_title(r'$\beta$ (in degrees)', fontsize = 20.0)
    axs[2].hist(e[mask], label = 'e')
    axs[2].set_title('e (in mm), h: ' + str(thickness) + str(r', $\delta$: '+str(deployment_ratio)+'h'), fontsize = 20.0)
    plt.show()

def plot_rigidity_angles(umesh):
    normal_angle, avg_angle, max_angle = compute_plate_normal_variation(umesh)
    mask = get_plate_center_joint_mask(umesh)
    fig, axs = plt.subplots(1, 2, figsize = (30, 10))
    axs[0].hist(avg_angle[mask]*180/np.pi)
    axs[0].set_title(r'Average variation in angle(degrees) of normals on plate', fontsize = 20.0)
    axs[1].hist(max_angle[mask]*180/np.pi, label = 'beta')
    axs[1].set_title(r'Maximum variation in angle(degrees) of normals on plate', fontsize = 20.0)
    plt.show()
    


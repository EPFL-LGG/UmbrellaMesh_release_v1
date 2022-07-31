from audioop import add
import json, gzip
from nbformat import read
import numpy as np
import numpy.linalg as la
import umbrella_mesh
import matplotlib.pyplot as plt


def read_data(filepath, material_params = [1400, 0.35, None, None, 14000, 0.35, None, None], handleBoundary = False, handlePivots = True):
    input_data = json.load(gzip.open(filepath))
    
    # Global config Data
    ## plate_edge_length
    ## arm_plate_edge_offset
    ## arm_joint_offset
    ## margin_length

    # Vis Data
    input_data['color'] = np.array(input_data['flip_bits'], dtype = np.bool_)
    input_data['base_mesh_v'] = np.array(input_data['base_mesh_v'])
    input_data['base_mesh_f'] = np.array(input_data['base_mesh_f'], dtype=np.int32)

    ## v_labels
    ## e_labels


    # Sim Data
    input_data['vertices'] = np.array(input_data['vertices'])
    input_data['edges'] = np.array(input_data['edges'], dtype=np.int32)
    input_data['ghost_bisectors'] = np.array(input_data['ghost_bisectors'])
    input_data['ghost_normals'] = np.array(input_data['ghost_normals'])
    input_data['segment_normals'] = np.array(input_data['segment_normals'])
    input_data['joint_type'] = []
    for vid, v_label in enumerate(input_data['v_labels']):
        if v_label[0] == 'O': # Rigid joint
            # print(input_data['vertices'][vid][2], input_data['vertices'][vid][2]*input_data['bbox_diagonal'])
            input_data['joint_type'].append(umbrella_mesh.JointType.Rigid)
            assert input_data['is_rigid'][vid] 
        elif v_label[0] == 'P': # T-joint
            if len(input_data['A_segments'][vid]) == 0 or len(input_data['B_segments'][vid]) == 0: # Rigid joint
                input_data['joint_type'].append(umbrella_mesh.JointType.Rigid)    
            else:
                input_data['joint_type'].append(umbrella_mesh.JointType.T)
            assert not input_data['is_rigid'][vid]
        elif v_label[0] == 'A' : # X-joint
            input_data['joint_type'].append(umbrella_mesh.JointType.X)
            assert not input_data['is_rigid'][vid]
        else: assert 0
    
    input_data['segment_type'] = []
    for eid, e_label in enumerate(input_data['e_labels']):
        if e_label[0] == 'P': # Plate
            input_data['segment_type'].append(umbrella_mesh.SegmentType.Plate)
        elif e_label == 'TA' or e_label == 'ANB': # Arm
            input_data['segment_type'].append(umbrella_mesh.SegmentType.Arm)
        else: assert 0
       
    
    ## uid
    for idx, uid_list in enumerate(input_data['uid']):
        if not isinstance(uid_list, list):
            input_data['uid'][idx] = [input_data['uid'][idx]]
        else:
            assert len(uid_list) == 2
            assert uid_list[0] != -1
            if uid_list[1] == -1:
                input_data['uid'][idx] = [uid_list[0]]

    # Uid - Top-Bot Joint Map
    num_uids = len(input_data['base_mesh_f'])
    uidTopBotMap = [[-1, -1] for _ in range(num_uids)]
    for vid, uid in enumerate(input_data['uid']):
        if input_data['v_labels'][vid] == 'OT':
            assert len(uid) == 1
            uidTopBotMap[uid[0]][0] = vid
        if input_data['v_labels'][vid] == 'OB':
            assert len(uid) == 1
            uidTopBotMap[uid[0]][1] = vid
    for joints in uidTopBotMap:
        assert -1 not in joints
    input_data['uid_top_bot_map'] = np.array(uidTopBotMap, dtype=np.int32)


    # umbrella connectivity
    input_data['umbrella_connectivity'] = []
    for fid1, face1 in enumerate(input_data['base_mesh_f']):
        for fid2, face2 in enumerate(input_data['base_mesh_f']):
            if fid2 <= fid1: continue
            if len(list(set(face1) & set(face2))) == 2: 
                input_data['umbrella_connectivity'].append([fid1, fid2])

    for i in range(len(input_data['midpoint_offsets_A'])):
        for j in range(len(input_data['midpoint_offsets_A'][i])):
            input_data['midpoint_offsets_A'][i][j] = np.array(input_data['midpoint_offsets_A'][i][j])
    for i in range(len(input_data['midpoint_offsets_B'])):
        for j in range(len(input_data['midpoint_offsets_B'][i])):
            input_data['midpoint_offsets_B'][i][j] = np.array(input_data['midpoint_offsets_B'][i][j])

    # correspondences
    for cid, corr in enumerate(input_data['correspondence']):
        if corr == None:
            assert input_data['joint_type'][cid] != umbrella_mesh.JointType.X
            input_data['correspondence'][cid] = np.array([0.0, 0.0, 0.0])
        else:
            assert  input_data['joint_type'][cid] == umbrella_mesh.JointType.X
            input_data['correspondence'][cid] = np.array(corr)
    input_data['correspondence'] = np.array(input_data['correspondence'])

    E1, nu1, t1, w1, E2, nu2, t2, w2 = material_params
    if t1 == None and 'thickness' in input_data: t1 = input_data['thickness']
    if t2 == None and 'thickness' in input_data: t2 = input_data['thickness']
    if t1 == None: t1 = 3.0/input_data['bbox_diagonal']
    if t2 == None: t2 = 3.0/input_data['bbox_diagonal']
    
    if w1 == None and 'width' in input_data: w1 = input_data['width']
    if w2 == None and 'width' in input_data: w2 = input_data['width']
    if w1 == None: w1 = 5.0/input_data['bbox_diagonal']
    if w2 == None: w2 = 5.0/input_data['bbox_diagonal']


    if handleBoundary:
        input_data = add_boundary_arms(input_data, handlePivots = handlePivots)
        
    
    ################################################################################
    # Transform the input data into an UmbrellaMeshIO structure
    ################################################################################
    joints = [umbrella_mesh.UmbrellaMeshIO.Joint(t, p, b, n, a, uid, corr) for t, p, b, n, a, uid, corr in
                                             zip(input_data['joint_type'],
                                                 input_data['vertices'],
                                                 input_data['ghost_bisectors'],
                                                 input_data['ghost_normals'],
                                                 input_data['alphas'],
                                                 input_data['uid'],
                                                 input_data['correspondence'])]
    umbrellas = []
    for uid in range(len(input_data['uid_top_bot_map'])):
        tj, bj, corr = *input_data['uid_top_bot_map'][uid], input_data['plate_correspondence'][uid]
        umbrellas.append(umbrella_mesh.UmbrellaMeshIO.Umbrella(tj, bj, corr))
    
    # Collect segment's endpoint joint connection data (joint, is_A, offset tuples)
    segment_endpoint_data = [[] for si in range(len(input_data['edges']))]
    for ji in range(len(input_data['vertices'])):
        for AB in 'AB':
            for s, o in zip(input_data[AB + '_segments'][ji], input_data['midpoint_offsets_' + AB][ji]):
                segment_endpoint_data[s].append((ji, AB == 'A', o))
    segments = [umbrella_mesh.UmbrellaMeshIO.Segment(t, [umbrella_mesh.UmbrellaMeshIO.JointConnection(*data) for data in endpoints], n) for t, endpoints, n in zip(input_data['segment_type'], segment_endpoint_data, input_data['segment_normals'])]
    io = umbrella_mesh.UmbrellaMeshIO(joints, segments, umbrellas, input_data['umbrella_connectivity'], [E1, nu1, t1, w1, E2, nu2, t2, w2], input_data['target_v'], input_data['target_f'])
    io.validate()

    return input_data, io

def nbrMapFromEdges(edges, numUmbrellas):
    nbr_map = [[] for _ in range(numUmbrellas)]
    for (e1, e2) in edges:
        nbr_map[e1].append(e2)
        nbr_map[e2].append(e1)
    return nbr_map

def get_cross_section(rod, index):
    m = rod.material(index)
    [a1, a2, a3, a4] = m.crossSectionBoundaryPts
    width = la.norm(a1 - a4)
    thickness = la.norm(a1 - a2)
    return np.array([width, thickness])
def normalize(vec):
        return vec / la.norm(vec)



def update_optimized_json(input_json_path, optim_heights_unscaled, optim_spacing_factor, output_json_path, handleBoundary = False, handlePivots = True):
    input_data = json.load(gzip.open(input_json_path))
    if 'umbrella_connectivity' not in input_data.keys(): # to check if the boundary has been processed
        input_data, _ = read_data(input_json_path, handleBoundary = handleBoundary, handlePivots = handlePivots)
    input_data['optim_heights'] = (optim_heights_unscaled * input_data['bbox_diagonal']).tolist()

    
    
    # umbrella connectivity
    input_data['umbrella_connectivity'] = []
    for fid1, face1 in enumerate(input_data['base_mesh_f']):
        for fid2, face2 in enumerate(input_data['base_mesh_f']):
            if fid2 <= fid1: continue
            if len(list(set(face1) & set(face2))) == 2: 
                input_data['umbrella_connectivity'].append([fid1, fid2])
                
    # Data for Tim - 3D Printing
    nbr_map = nbrMapFromEdges(input_data['umbrella_connectivity'], len(input_data['flip_bits']))
    tim_data = {'top_heights': [None for _ in range(len(nbr_map))],
                'bot_heights': [None for _ in range(len(nbr_map))],
                'connectivity': nbr_map,
                'flip_bits': input_data['flip_bits'],
                'target_spacing': optim_spacing_factor*input_data['thickness'],
                'umbrella_center': [None for _ in range(len(nbr_map))],
                'bbox_diagonal': input_data['bbox_diagonal']
                }
    # Heights: Centerline of X compliant joint to plate centerline
    # top_heights and bot_heights follow an umbrella ordering upon which the connectivity array is built. For each umbrella index uid, connectivity[uid] gives its neighbors.
    # flip_bits: a value of 1 means, at rest state, the umbrella has the compliant joints closer than farther to its central axis. 
    # umbrella_center: [x, y] coordinates of the center of the top/bot triangle of umbrella uid is found in umbrella_center[uid]

    for vid, v in enumerate(input_data['vertices']):
        if input_data['v_labels'][vid] == 'OT':
            assert tim_data['top_heights'][input_data['uid'][vid][0]] == None
            tim_data['top_heights'][input_data['uid'][vid][0]] = v[2]
            tim_data['umbrella_center'][input_data['uid'][vid][0]] = v[:2]
        if input_data['v_labels'][vid] == 'OB':
            assert tim_data['bot_heights'][input_data['uid'][vid][0]] == None
            tim_data['bot_heights'][input_data['uid'][vid][0]] = -v[2]
    
    for cid, c in enumerate(tim_data['umbrella_center']):
        tim_data['umbrella_center'][cid] = (input_data['bbox_diagonal']* np.array(tim_data['umbrella_center'][cid])).tolist()
    if 'optim_heights' in input_data:
        num_uids = len(input_data['optim_heights'])//2
        tim_data['top_heights'] = input_data['optim_heights'][:num_uids]#*input['bbox_diagonal']
        tim_data['bot_heights'] = input_data['optim_heights'][num_uids:]#*input['bbox_diagonal']
    
    assert not (None in tim_data['top_heights'])
    assert not (None in tim_data['bot_heights'])
    assert not (None in tim_data['umbrella_center'])


    input_data = json_serializable(input_data)
    with gzip.open(output_json_path, 'wt') as outfile:
        json.dump(input_data, outfile, indent=4)

    with open(output_json_path[:-8] + "_tim.json", 'w') as outfile:
        json.dump(tim_data, outfile, indent=4)


def write_deformed_config(curr_um, input_path, output_path, write_stress = False, is_rest_state = False, handleBoundary = False, handlePivots = True, reg_data = None):
    points = []
    normals = []
    cross_sections = []
    bending_stresses = []
    twisting_stresses = []
    input_data = json.load(gzip.open(input_path))
    if 'umbrella_connectivity' not in input_data.keys(): # to check if the boundary has been processed
        input_data, _ = read_data(input_path, handleBoundary = handleBoundary, handlePivots = handlePivots)
    
    if write_stress:
        bending_stress = curr_um.maxBendingStresses()
        twisting_stress = curr_um.twistingStresses()


    if is_rest_state == False:
        tsf = curr_um.getTargetSurface()
        points_umbrella = tsf.getQueryPtPos(curr_um).reshape(-1,3)
        tsf.forceUpdateClosestPoints(curr_um)
        points_target = tsf.umbrella_closest_surf_pts.reshape(-1,3)

        distance = np.linalg.norm(points_umbrella - points_target, axis = 1)
        max_distance = distance.max()
        distance /= max_distance
        cmap = plt.cm.PuRd
        colors = cmap(distance)

        input_data['tf_max_distance'] = max_distance
        input_data['tf_points_umbrella'] = points_umbrella.tolist()
        input_data['tf_points_target'] = points_target.tolist()
        input_data['tf_colors'] = colors.tolist()
        input_data['tf_is_point_bdry'] = curr_um.IsQueryPtBoundary()




    physical_joints = [[] for _ in range(curr_um.numJoints())]
    physical_joint_normals = [[] for _ in range(curr_um.numJoints())]
    for seg_id in range(curr_um.numSegments()):
        curr_seg = curr_um.segment(seg_id)
    
    for jid in range(curr_um.numJoints()):
        curr_j  = curr_um.joint(jid)
        physical_joint_normals[jid] = curr_j.ghost_normal().tolist()
        physical_joints[jid] = curr_j.position.tolist()

    for seg_id in range(curr_um.numSegments()):
        curr_seg = curr_um.segment(seg_id)
        curr_rod = curr_seg.rod
        dc = curr_rod.deformedConfiguration()
        assert curr_seg.startJoint < curr_um.numJoints() and curr_seg.endJoint < curr_um.numJoints()
        curr_points = curr_rod.deformedPoints()
        for pid, point in enumerate(curr_points):
            curr_points[pid] = point.tolist()
        curr_normals = []
        curr_cross_sections = []
        curr_bending_stresses = []
        curr_twisting_stresses = []
        curr_normals.append(dc.materialFrame[0].d2.tolist())
        curr_cross_sections.append(get_cross_section(curr_rod, 0).tolist())
        for edge_index in range(len(dc.materialFrame))[1:]:
            curr_normals.append(normalize(dc.materialFrame[edge_index-1].d2 + dc.materialFrame[edge_index].d2).tolist())
            curr_cross_sections.append(((get_cross_section(curr_rod, edge_index-1) + get_cross_section(curr_rod, edge_index))/2.0).tolist())
        curr_normals.append(dc.materialFrame[-1].d2.tolist())
        curr_cross_sections.append(get_cross_section(curr_rod, curr_rod.numEdges() - 1).tolist())
        if write_stress:
            curr_bending_stresses.append(bending_stress[seg_id].tolist())
            curr_twisting_stresses.append(twisting_stress[seg_id].tolist())
        
        if curr_seg.startJoint == input_data['edges'][seg_id][1]:
            curr_points.reverse()
            curr_normals.reverse()
            curr_cross_sections.reverse()
            if write_stress:
                curr_bending_stresses.reverse()
                curr_twisting_stresses.reverse()
        else:
            assert curr_seg.startJoint == input_data['edges'][seg_id][0]
            
        points.append(curr_points)
        normals.append(curr_normals)
        cross_sections.append(curr_cross_sections)
        if write_stress:
            bending_stresses.append(curr_bending_stresses)
            twisting_stresses.append(curr_twisting_stresses)
    if is_rest_state:
        input_data['rest_subdiv_points'] = points
        input_data['rest_normals'] = normals
        input_data['rest_cross_sections'] = cross_sections
        input_data['rest_physical_joints'] = physical_joints
        input_data['rest_physical_joint_normals'] = physical_joint_normals
        if write_stress:
            input_data['rest_bending_stresses'] = bending_stresses
            input_data['rest_twisting_stresses'] = twisting_stresses
    else:
        input_data['deformed_subdiv_points'] = points
        input_data['deformed_normals'] = normals
        input_data['deformed_cross_sections'] = cross_sections
        input_data['deformed_physical_joints'] = physical_joints
        input_data['deformed_physical_joint_normals'] = physical_joint_normals
        if write_stress:
            input_data['deformed_bending_stresses'] = bending_stresses
            input_data['deformed_twisting_stresses'] = twisting_stresses
            
    
    if reg_data is not None:
        assert len(reg_data) == 2
        input_data['reg_R'] = reg_data[0]
        input_data['reg_t'] = reg_data[1]
    input_data = json_serializable(input_data)
    
    with gzip.open(output_path, 'wt') as outfile:
        json.dump(input_data, outfile, indent=4)


def json_serializable(input_data):

    def serialize_list(in_list):
        for id, entry in enumerate(in_list):
            if type(entry) == np.ndarray:
                in_list[id] = entry.tolist()
            elif type(entry) == list:
                in_list[id] = serialize_list(entry)
        return in_list
    if 'joint_type' in input_data.keys(): del input_data['joint_type']
    if 'segment_type' in input_data.keys(): del input_data['segment_type']
    for key in input_data.keys():
        if type(input_data[key]) == np.ndarray:
            input_data[key] = input_data[key].tolist()
        if type(input_data[key]) == list:
            input_data[key] = serialize_list(input_data[key])

    return input_data
            

def add_boundary_arms(input_data, handlePivots = True):
    for vid1, v in enumerate(input_data['vertices']):
        if input_data['v_labels'][vid1] == 'PT' and len(input_data['midpoint_offsets_B'][vid1]) == 0:# Doesn't have an arm :(
            uid = input_data['uid'][vid1][0]
            for vid2, v in enumerate(input_data['vertices']):
                if input_data['uid'][vid2][0] == uid and input_data['v_labels'][vid2] == 'PB' and len(input_data['midpoint_offsets_B'][vid2]) == 0:# Doesn't have an arm :(
                    # Connect these two with an Arm!
                    # if handlingPivots, lower PT joint and lift PB  appropriately
                    seg_normal_dir = np.cross(input_data['ghost_normals'][vid1], input_data['ghost_bisectors'][vid1])
                    seg_normal_dir /= np.linalg.norm(seg_normal_dir)
                    
                    if handlePivots:
                        input_data['vertices'][vid1] +=  np.array([0,0,-1.0])*input_data['thickness']/2
                        input_data['vertices'][vid2] +=  np.array([0,0,1.0])*input_data['thickness']/2

                        input_data['vertices'][vid1] -=  seg_normal_dir*input_data['thickness']/2
                        input_data['vertices'][vid2] -=  seg_normal_dir*input_data['thickness']/2
                    x1, y1, z = input_data['vertices'][vid1].tolist()
                    x2, y2, z = input_data['vertices'][vid2].tolist()
                    
                    if input_data['flip_bits'][uid]:
                        input_data['vertices'] = np.concatenate((input_data['vertices'], np.array([[0.5*(x1 + x2), 0.5*(y1+y2), 0.0]])))
                    else:
                        input_data['vertices'] = np.concatenate((input_data['vertices'], np.array([[0.5*(x1 + x2), 0.5*(y1+y2), 0.0]]) + seg_normal_dir*input_data['thickness']))
                    input_data['v_labels'].append('AAM')

                    input_data['uid'].append([uid])
                    input_data['edges'] = np.concatenate((input_data['edges'], np.array([[vid1, len(input_data['vertices'])-1]]))) 
                    input_data['edges'] = np.concatenate((input_data['edges'], np.array([[len(input_data['vertices'])-1, vid2]])))
                    input_data['e_labels'].append('TA')
                    input_data['e_labels'].append('ANB')
                    input_data['segment_normals'] = np.concatenate((input_data['segment_normals'], seg_normal_dir[np.newaxis, ...]))
                    input_data['segment_normals'] = np.concatenate((input_data['segment_normals'], -seg_normal_dir[np.newaxis, ...]))
                    input_data['segment_type'].append(umbrella_mesh.SegmentType.Arm)
                    input_data['segment_type'].append(umbrella_mesh.SegmentType.Arm)

                    
                    eid1, eid2 = len(input_data['edges'])-2, len(input_data['edges'])-1
                    input_data['B_segments'][vid1].append(eid1)
                    input_data['B_segments'][vid2].append(eid2)
                    input_data['midpoint_offsets_B'][vid1].append(seg_normal_dir*input_data['thickness']/2)
                    input_data['midpoint_offsets_B'][vid2].append(seg_normal_dir*input_data['thickness']/2)
                    
                    assert len(input_data['midpoint_offsets_A'][vid1]) == 1 and np.linalg.norm(input_data['midpoint_offsets_A'][vid1][0]) == 0
                    assert len(input_data['midpoint_offsets_A'][vid2]) == 1 and np.linalg.norm(input_data['midpoint_offsets_A'][vid2][0]) == 0
                    input_data['midpoint_offsets_A'][vid1][0] = np.array([0,0,1.0])*input_data['thickness']/2
                    input_data['midpoint_offsets_A'][vid2][0] = np.array([0,0,-1.0])*input_data['thickness']/2
                    
                    input_data['is_rigid'][vid1] = False
                    input_data['is_rigid'][vid2] = False
                    input_data['joint_type'][vid1] = umbrella_mesh.JointType.T
                    input_data['joint_type'][vid2] = umbrella_mesh.JointType.T

                    # Details of vid_new
                    vid_new = len(input_data['vertices']) - 1
                    input_data['ghost_normals'] = np.concatenate((input_data['ghost_normals'], -input_data['ghost_normals'][vid1][np.newaxis, ...]))
                    input_data['ghost_bisectors'] = np.concatenate((input_data['ghost_bisectors'], input_data['ghost_bisectors'][vid1][np.newaxis, ...]))
                    input_data['A_segments'].append([eid1]) 
                    input_data['B_segments'].append([eid2])

                    input_data['is_rigid'].append(False)
                    input_data['alphas'].append(0)
                    corr = input_data['boundary_correspondence'][uid]
                    input_data['correspondence'] = np.concatenate((input_data['correspondence'], np.array(corr)[np.newaxis, ...]))
                    input_data['joint_type'].append(umbrella_mesh.JointType.X)

                    thickness = input_data['thickness']  
                    if input_data['flip_bits'][uid]:
                        input_data['midpoint_offsets_A'].append([seg_normal_dir*thickness/2 + input_data['ghost_normals'][vid1]*-1*input_data['arm_joint_offset']])
                        input_data['midpoint_offsets_B'].append([seg_normal_dir*thickness/2 + input_data['ghost_normals'][vid2]*-1*input_data['arm_joint_offset']])
                    else:
                        input_data['midpoint_offsets_A'].append([-seg_normal_dir*thickness/2 + input_data['ghost_normals'][vid1]*-1*input_data['arm_joint_offset']])
                        input_data['midpoint_offsets_B'].append([-seg_normal_dir*thickness/2 + input_data['ghost_normals'][vid2]*-1*input_data['arm_joint_offset']])
                    
    return input_data

#include "UmbrellaMesh.hh"
#include "UmbrellaMeshTerminalEdgeSensitivity.hh"
#include "UmbrellaTargetSurfaceFitter.hh"

#include <algorithm>
#include <iterator>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/utils.hh>
#include <MeshFEM/filters/merge_duplicate_vertices.hh>
#include <MeshFEM/Geometry.hh>
#if MESHFEM_WITH_TBB
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#endif

#define ENABLE_TBB 1

#include <Eigen/Eigenvalues>

#include <MeshFEM/unused.hh>
#include <MeshFEM/ParallelAssembly.hh>


template<typename Real_>
void UmbrellaMesh_T<Real_>::set(const UmbrellaMeshIO &io, size_t subdivision) {
    if (subdivision < 5) throw std::runtime_error("Rods in an umbrella mesh must have at least 5 edges (to prevent conflicting start/end joint constraints and fully separate joint influences in Hessian)");

    io.validate();
    const size_t nj = io.joints.size();
    const size_t ns = io.segments.size();
    const size_t nu = io.umbrellas.size();

    m_segments.clear();
    m_joints.clear();
    m_segments.reserve(ns);
    m_joints.reserve(nj);
    m_numRigidJoints = 0;

    // Collect the terminal edge information needed to construct each joint.
    std::vector<std::vector<typename Joint::TerminalEdgeInputData>> terminalEdgeInputs(nj);

    ////////////////////////////////////////////////////////////////////////////
    // Construct the segments
    ////////////////////////////////////////////////////////////////////////////
    m_numArmSegments = 0;
    m_armIndexForSegment.resize(ns);
    m_segmentIndexForArm.clear();
    VecX segmentRestLenGuess(ns);
    for (size_t si = 0; si < ns; ++si) {
        m_armIndexForSegment[si] = m_numArmSegments;
        const auto &s = io.segments[si];
        if (s.type == SegmentType::Arm) {
            m_segmentIndexForArm.push_back(si);
            ++m_numArmSegments;
        }

        int j0 = s.endpoint[0].joint_index,
            j1 = s.endpoint[1].joint_index;
        Vec3 p0 = io.joints[j0].position + s.endpoint[0].midpoint_offset,
             p1 = io.joints[j1].position + s.endpoint[1].midpoint_offset;
        Vec3 t = p1 - p0;
        Real_ len = t.norm();
        t.normalize();

        // Create segment
        m_segments.emplace_back(p0, p1, subdivision, s.type);
        m_segments[si].startJoint = j0;
        m_segments[si].  endJoint = j1;
        segmentRestLenGuess[si] = len;

        // Accumulate terminal endpoint information
        // Note: the segment's tangent is used for both the start and the end joint (never its negation)!
        terminalEdgeInputs[j0].emplace_back(si, m_segments[si].rod.restLengths()[              0], t, s.normal.normalized(), s.endpoint[0].midpoint_offset, s.endpoint[0].is_A,  true);
        terminalEdgeInputs[j1].emplace_back(si, m_segments[si].rod.restLengths()[subdivision - 1], t, s.normal.normalized(), s.endpoint[1].midpoint_offset, s.endpoint[1].is_A, false);

        if (std::abs(stripAutoDiff(t.dot(s.normal.normalized()))) > 1e-8)
            throw std::runtime_error("Segment " + std::to_string(si) + "'s tangent not perpendicular to normal");

    }

    

    ////////////////////////////////////////////////////////////////////////////
    // Link the umbrellas
    // TODO: update members to look more like UmbrellaMeshIO...
    ////////////////////////////////////////////////////////////////////////////
    m_umbrella_to_top_bottom_joint_map.resize(nu);
    m_umbrella_connectivity = io.umbrella_connectivity;
    for (size_t ui = 0; ui < nu; ++ui) {
        const auto &u = io.umbrellas[ui];
        m_umbrella_to_top_bottom_joint_map[ui] = {{ u.top_joint, u.bottom_joint }};
        m_umbrella_tgt_pos.push_back(u.tgt_pos);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Construct the joints
    ////////////////////////////////////////////////////////////////////////////
    for (size_t ji = 0; ji < nj; ++ji) {
        const auto &j = io.joints[ji];
        if (j.bisector.cross(j.normal).norm() < 1e-8) std::cerr<<"At joint "<<ji<<", input source vector is parallel to input normal!"<<std::endl;
        m_joints.emplace_back(this, j.position, j.alpha, j.normal.normalized(), j.bisector.normalized(),
                              terminalEdgeInputs[ji], j.type, j.umbrella_ID);
        if (j.type == JointType::Rigid) ++m_numRigidJoints;
        if (j.type == JointType::    X) {
            m_X_joint_indices.push_back(ji);
            m_X_joint_tgt_pos.push_back(j.tgt_pos);
        }
    }

    std::vector<size_t> umbrellaArmSegmentCounter(nu, 0);
    // Asuming arm joints always have four segments. 2 in A, 2 in B.
    for (size_t ji = 0; ji < nj; ++ji) {
        if (io.joints[ji].type != JointType::X) continue;

        if(terminalEdgeInputs[ji].size() != 4) continue; //throw std::runtime_error("Arm joint with number of segments not equal to 4");
        std::vector<std::pair<size_t, size_t> > A_ends, B_ends;
        for (auto &tei : terminalEdgeInputs[ji]) {
            size_t njid = tei.isStart ? m_segments[tei.si].endJoint : m_segments[tei.si].startJoint;
            if(tei.is_A) A_ends.push_back(std::pair<size_t, size_t>(njid, tei.si));
            else B_ends.push_back(std::pair<size_t, size_t>(njid, tei.si));
        }
        if(A_ends.size() != 2 || B_ends.size() != 2) throw std::runtime_error("Assumption violated");
        if(m_joints[A_ends[0].first].umbrellaID() != m_joints[B_ends[0].first].umbrellaID()) {
            std::reverse(B_ends.begin(), B_ends.end());
        }
        if(m_joints[A_ends[0].first].umbrellaID() != m_joints[B_ends[0].first].umbrellaID()) throw std::runtime_error("There should only be two unique UID vecs among these four end points");

        for (size_t eid = 0; eid < A_ends.size(); ++eid) {
            std::pair<size_t, size_t> A_end = A_ends[eid], B_end = B_ends[eid];
            size_t uid = m_joints[A_end.first].umbrellaID()[0];
            m_segments[A_end.second].setArmSegmentPosType(static_cast<ArmSegmentPosType>(umbrellaArmSegmentCounter[uid]));
            m_segments[B_end.second].setArmSegmentPosType(static_cast<ArmSegmentPosType>(umbrellaArmSegmentCounter[uid]++));
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Initialize other parameters and derived quantities.
    ////////////////////////////////////////////////////////////////////////////
    // Build the segment len->edge len map and use it to construct
    // the rest length guess for every edge in the network
    m_perArmRestLen.resize(m_numArmSegments);
    for (size_t ai = 0; ai < m_numArmSegments; ++ai)
        m_perArmRestLen[ai] = segmentRestLenGuess[m_segmentIndexForArm[ai]];

    m_numURH = 2 * nu;

    m_targetDeploymentHeight = VecX::Zero(nu);
    
    m_constructArmRestLenToEdgeRestLenMapTranspose();
    // Since the perArmRestLen and Umbrella Heights are consistent at initialization, it doesn't matter which set rest lengths is called.
    m_setRestLengthsFromPARL();

    // Default is to use only rest length.
    setDesignParameterConfig(true, false, true);
    
    // Initialize DoF offset table.
    m_buildDoFOffsets();

    auto params = getDoFs();
    setDoFs(params, true /* set spatially coherent thetas */);

    // The terminal edges of each segment have been twisted to conform to
    // the joints, but the internal edges are in their default orientation.
    // We update the edges' material axes (thetas) by minimizing the twist
    // energy with respect to the thetas.
    for (auto &s : m_segments)
        s.setMinimalTwistThetas();

    // Update the "source thetas" used to maintain temporal coherence
    updateSourceFrame();

    // Load material from IO
    
    Real_ E1 = io.material_params[0], nu1 = io.material_params[1];
    Real_ E2 = io.material_params[4], nu2 = io.material_params[5];
    std::vector<double>   armCrossSection = {io.material_params[2], io.material_params[3]};
    std::vector<double> plateCrossSection = {io.material_params[6], io.material_params[7]};

    // Note: the material object must retain the cross-section mesh to enable stress analysis.
      m_armMaterial = RodMaterial(std::string("rectangle"), stripAutoDiff(E1), stripAutoDiff(nu1), stripAutoDiff(  armCrossSection), RodMaterial::StiffAxis::D1, /* keepCrossSectionMesh = */ true);
    m_plateMaterial = RodMaterial(std::string("rectangle"), stripAutoDiff(E2), stripAutoDiff(nu2), stripAutoDiff(plateCrossSection), RodMaterial::StiffAxis::D1, /* keepCrossSectionMesh = */ true);
      m_armMaterial.stressAnalysis(); // Ensure the   arm material stress analysis object is cached/shared across all rods
    m_plateMaterial.stressAnalysis(); // Ensure the plate material stress analysis object is cached/shared across all rods
    setMaterial(m_armMaterial, m_plateMaterial);
    m_initMinRestLen = minRestLength();

    // Initialize the deployment weight vector.
    setUniformDeploymentEnergyWeight(m_uniformDeploymentEnergyWeight);

    // Set opposite center for joints.
    setOppositeCenter();

    m_clearCache();

    set_target_surface(io.target_v, io.target_f);
    {
        Vec3 bbMin, bbMax;
        std::vector<Pt3> pts = deformedPoints();
        bbMin = bbMax = pts[0];
        for (const auto &p : pts) {
            bbMin = bbMin.cwiseMin(p);
            bbMax = bbMax.cwiseMax(p);
        }
        m_l0 = autodiffCast<Real>((bbMax - bbMin).norm());
    }
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::m_buildDoFOffsets() {
    m_dofOffsetForSegment.resize(m_segments.size());
    m_dofOffsetForJoint.resize(m_joints.size());

    m_dofOffsetForCenterlinePos.clear();
    m_dofOffsetForCenterlinePos.reserve(numSegments() * m_segments.front().rod.numVertices());

    size_t offset = 0;
    for (size_t i = 0; i < numSegments(); ++i) {
        m_dofOffsetForSegment[i] = offset;
        for (size_t j = 0; j < m_segments[i].numFreeVertices(); ++j)
            m_dofOffsetForCenterlinePos.push_back(offset + 3 * j);
        offset += m_segments[i].numDoF();
    }
    for (size_t i = 0; i < numJoints(); ++i) {
        m_dofOffsetForJoint[i] = offset;
        offset += m_joints[i].numDoF();
    }

    // Compute rest length offset.
    m_restLenDofOffsetForSegment.resize(numSegments());
    for (size_t i = 0; i < numSegments(); ++i) {
        m_restLenDofOffsetForSegment[i] = offset;
        if (m_segments[i].segmentType() == SegmentType::Arm)
            offset += m_segments[i].numFreeEdges() * m_umbrella_dPC.restLen;
    }
    // Currently the design parameter for joint only has rest length. 
    m_designParameterDoFOffsetForJoint.resize(m_joints.size());
    for (size_t i = 0; i < numJoints(); ++i) {
        m_designParameterDoFOffsetForJoint[i] = offset;
        offset += m_joints[i].numArms() * m_umbrella_dPC.restLen;
    }
}


// Construct the *transpose* of the map from a vector holding the (rest) lengths
// of each segment to a vector holding a (rest) length for every *variable* rod length in the
// entire network. The vector output by this map is ordered as follows: all
// lengths for segments' interior and free edges, followed by the length variables for each joint.
// (We use build the transpose instead of the map itself to efficiently support
// the iteration needed to assemble the Hessian chain rule term)
// This is a fixed linear map for the lifetime of the umbrella mesh: the segment length is evenly
// distributed to each unconstrained discrete edges of each variable rod. 
template<typename Real_>
void UmbrellaMesh_T<Real_>::m_constructArmRestLenToEdgeRestLenMapTranspose() {
    // Get the initial ideal rest length for the edges of each segment; this is
    // used to decide which segments control which terminal edges.
    std::vector<size_t> numEdgesForArmSegment(numArmSegments());
    size_t counter = 0;
    size_t totalFreeEdges = 0;
    for (size_t si = 0; si < numSegments(); ++si) {
        if (segment(si).segmentType() == SegmentType::Arm) {
            numEdgesForArmSegment[counter] = segment(si).rod.numEdges();
            totalFreeEdges += numEdgesForArmSegment[counter] - segment(si).hasStartJoint() - segment(si).hasEndJoint();
            counter += 1;
        }
    }

    // Decide who controls each joint edge: the shorter ideal rest length
    // wins. Ties are broken arbitrarily.
    std::vector<std::vector<size_t>> incidentArmsForJoint(numJoints());
    for (size_t ji = 0; ji < numJoints(); ++ji) {
        const auto &j = joint(ji);
        auto &c = incidentArmsForJoint[ji];
        c.clear();
        for (size_t si = 0; si < j.valence(); ++si) {
            if (j.getSegmentTypeAt(si) == SegmentType::Arm)
                c.push_back(getArmIndexAt(j.getSegmentAt(si)));
        }
    }

    // Determine the number of nonzeros in the map.
    // Each free edge in a segment segment is potentially influenced by segment
    // lengths in the stencil:
    //      +-----+-----+-----+
    //               ^
    // (The segment always influences its own edges.

    // Each arm of the joint will have a single entry.
    size_t totalJointArms = 0;
    for (auto & j : m_joints) {
        totalJointArms += j.numArms();
    }
    size_t nz = totalFreeEdges + totalJointArms;
    const SuiteSparse_long m = numArmSegments(), n = nz;
    SuiteSparseMatrix result(m, n);
    result.nz = nz;

    // Now we fill out the transpose of the map one column (edge) at a time:
    //    #     [               ]
    // segments [               ]
    //              # edges
    auto &Ai = result.Ai;
    auto &Ax = result.Ax;
    auto &Ap = result.Ap;

    Ai.reserve(nz);
    Ax.reserve(nz);
    Ap.reserve(n + 1);

    Ap.push_back(0); // col 0 begin

    // Segments are split into (ne - 1) intervals spanning between
    // the incident joint positions (graph nodes); half an interval
    // extends past the joints at each end.
    // Joints control the lengths of the intervals surrounding them,
    // specifying the length of half a subdivision interval on the incident
    // segments. The remaining length of each segment is then
    // distributed evenly across the "free" intervals.
    // First, build the columns for the free edges of each segment:

    for (size_t ai = 0; ai < numArmSegments(); ++ai) {
        const auto &s = segment(getSegmentIndexForArmAt(ai));
        auto nfe = numEdgesForArmSegment[ai] - s.hasStartJoint() - s.hasEndJoint();
        // Visit each internal/free edge:
        for (size_t ei = 0; ei < nfe; ++ei) {
            Ai.push_back(ai);
            Ax.push_back(1.0 / (numEdgesForArmSegment[ai] - 1));
            Ap.push_back(Ai.size()); // col end
        }
    }

    // Build the columns for the joint edges
    for (size_t ji = 0; ji < numJoints(); ++ji) {
        for (size_t ai = 0; ai < incidentArmsForJoint[ji].size(); ++ai) {
            const size_t c = incidentArmsForJoint[ji][ai];
            Ai.push_back(c);
            Ax.push_back(1.0 / (numEdgesForArmSegment[c] - 1));
            Ap.push_back(Ai.size()); // col end
        }
    }
    assert(Ax.size() == size_t(nz   ));
    assert(Ai.size() == size_t(nz   ));
    assert(Ap.size() == size_t(n + 1));

    m_armRestLenToEdgeRestLenMapTranspose = std::move(result);
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::setMaterial(const RodMaterial &armMat, const RodMaterial &plateMat) {
      m_armMaterial =   armMat;
    m_plateMaterial = plateMat;

    for (auto &s : m_segments)
        s.rod.setMaterial((s.segmentType() == SegmentType::Plate) ? plateMat : armMat);

    // Changing the material can change the cross-section, resulting in a
    // different normal offset magnitude at the joints.
    // We update the terminal edges accordingly by re-applying the joint
    // configuration:
    setDoFs(getDoFs(), false);
    updateSourceFrame();
}

template<typename Real_>
size_t UmbrellaMesh_T<Real_>::numDoF() const {
    size_t result = 0;
    for (const auto &s : m_segments) result += s.numDoF();
    for (const auto &j :   m_joints) result += j.numDoF();
    return result;
}

// Full parameters consist of all segment parameters followed by all joint parameters.
template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::getDoFs() const {
    VecX params(numDoF());

    parallel_for_range(numSegments(), [&](size_t i) { auto slice = params.segment(m_dofOffsetForSegment[i], m_segments[i].numDoF()); m_segments[i].getParameters(slice); });
    parallel_for_range(  numJoints(), [&](size_t i) { auto slice = params.segment(m_dofOffsetForJoint  [i], m_joints  [i].numDoF()); m_joints  [i].getParameters(slice); });

    return params;
}

// Full parameters consist of all segment parameters followed by all joint parameters.
// "spatialCoherence" affects how terminal edge thetas are determined from the
// joint parameters; see joint.applyConfiguration.
template<typename Real_>
void UmbrellaMesh_T<Real_>::setDoFs(const Eigen::Ref<const VecX> &params, bool spatialCoherence) {
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".setDoFs");
    const size_t n = numDoF();
    if (size_t(params.size()) != n) throw std::runtime_error("Invalid number of parameters");

    const size_t ns = m_segments.size();
    m_networkPoints.resize(ns);
    m_networkThetas.resize(ns);

    // First, unpack the segment parameters into the points/thetas arrays
    auto processSegment = [&](size_t si) {
        auto slice = params.segment(m_dofOffsetForSegment[si], m_segments[si].numDoF());
        m_segments[si].unpackParameters(slice, m_networkPoints[si], m_networkThetas[si]);
    };
    parallel_for_range(ns, processSegment);

    // Second, set all joint parameters and then
    // use them to configure the segments' terminal edges.
    const size_t nj = m_joints.size();
    auto processJoint = [&](size_t ji) {
        m_joints[ji].setParameters(params.segment(m_dofOffsetForJoint[ji], m_joints[ji].numDoF()));
        m_joints[ji].applyConfiguration(m_segments, m_networkPoints, m_networkThetas, spatialCoherence);
    };
    parallel_for_range(nj, processJoint);

    // Finally, set the deformed state of each rod in the network
    parallel_for_range(ns, [&](size_t si) { m_segments[si].rod.setDeformedConfiguration(m_networkPoints[si], m_networkThetas[si]); });

    m_sensitivityCache.clear();

    if (hasTargetSurface()) { m_target_surface_fitter->updateClosestPoints(*this); }
}

template<typename Real_>
size_t UmbrellaMesh_T<Real_>::numFreeRestLengths() const {
    size_t result = 0;
    // A rest length for every free (non-joint) edge of each segment.
    for (const auto &s : m_segments) {
        if (s.segmentType() == SegmentType::Arm) {
            result += s.numFreeEdges();
        }
    }
    return result;
}

template<typename Real_>
size_t UmbrellaMesh_T<Real_>::numJointRestLengths() const {
    size_t result = 0;
    for (const auto &j : m_joints) result += j.numArms();
    return result;
}

template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::getFreeRestLengths() const {
    VecX result(numFreeRestLengths());

    // A rest length for every free (non-joint) edge of each segment.
    size_t offset = 0;
    for (const auto &s : m_segments) {
        if (s.segmentType() == SegmentType::Arm) {
            auto rlens = s.rod.restLengths();
            const size_t nfe = s.numFreeEdges();
            result.segment(offset, nfe) = Eigen::Map<VecX>(rlens.data(), rlens.size()).segment(s.hasStartJoint(), nfe);
            offset += nfe;
        }
    }

    return result;
}

template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::getJointRestLengths() const {
    VecX result(numJointRestLengths());

    size_t offset = 0;
    for (const auto &j : m_joints) {
        result.segment(offset, j.numArms()) = j.getRestLengths();
        offset += j.numArms();
    }
    return result;
}

template<typename Real_>
size_t UmbrellaMesh_T<Real_>::numRestLengths() const { return numFreeRestLengths() + numJointRestLengths(); }

template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::getRestLengths() const {
    VecX result(numRestLengths());
    result.segment(0, numFreeRestLengths()) = getFreeRestLengths();
    result.segment(numFreeRestLengths(), numJointRestLengths()) = getJointRestLengths();
    return result;
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::m_setRestLengthsFromPARL() {
    if (m_armRestLenToEdgeRestLenMapTranspose.m == 0) throw std::runtime_error("Must run m_constructArmRestLenToEdgeRestLenMapTranspose first");
    VecX restLens = m_armRestLenToEdgeRestLenMapTranspose.apply(m_perArmRestLen, /* transpose */ true);
    // Apply these rest lengths to the linkage.
    size_t offset = 0;
    for (auto &s : m_segments) {
        if (s.segmentType() == SegmentType::Arm) {
            const size_t ne = s.rod.numEdges();
            // Visit each internal/free edge:
            for (size_t ei = s.hasStartJoint(); ei < (s.hasEndJoint() ? ne - 1 : ne); ++ei)
                s.rod.restLengthForEdge(ei) = restLens[offset++];
        }
    }
    for (auto &j : m_joints) {
        if (j.numArms() > 0) j.setRestLengths(restLens.segment(offset, j.numArms()));
        offset += j.numArms();
    }

    if (offset != size_t(restLens.size()))
        throw std::logic_error("Unexpected restLens size");
}

template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::getExtendedDoFs() const {
    VecX result(numExtendedDoF());
    result.segment(0, numDoF()) = getDoFs();
    if (m_umbrella_dPC.restLen) result.segment(numDoF(), numRestLengths()) = getRestLengths();
    return result;
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::setExtendedDoFs(const VecX_T<Real_> &params, bool spatialCoherence) {
    if (size_t(params.size()) != numExtendedDoF()) throw std::runtime_error("Extended DoF size mismatch");
    size_t offset = numDoF();
    setDoFs(params.segment(0, offset), spatialCoherence);

    if (m_umbrella_dPC.restLen) {
        // A rest length for every free (non-joint) edge of each segment.
        for (auto &s : m_segments) {
            if (s.segmentType() == SegmentType::Arm) {
                auto rlens = s.rod.restLengths();
                const size_t nfe = s.numFreeEdges();
                Eigen::Map<VecX>(rlens.data(), rlens.size()).segment(s.hasStartJoint(), nfe) = params.segment(offset, nfe);
                s.rod.setRestLengths(rlens);
                offset += nfe;
            }
        }
    }

    if (m_umbrella_dPC.restLen) {
        for (auto &j : m_joints) {
            if (j.numArms() > 0) j.setRestLengths(params.segment(offset, j.numArms()));
            offset += j.numArms();
        }
    }
    if (hasTargetSurface()) m_target_surface_fitter->updateClosestPoints(*this);
}

template<typename Real_>
Real_ UmbrellaMesh_T<Real_>::energyElastic() const {
    // BENCHMARK_SCOPED_TIMER_SECTION timer("energyElastic");
    return summation_parallel([&](size_t si) { return m_segments[si].rod.energy(); }, numSegments());
}

template<typename Real_>
Real_ UmbrellaMesh_T<Real_>::energyStretch() const {
    Real_ result = 0;
    for (const auto &s : m_segments) result += s.rod.energyStretch();
    return result;
}

template<typename Real_>
Real_ UmbrellaMesh_T<Real_>::energyBend() const {
    Real_ result = 0;
    for (const auto &s : m_segments) result += s.rod.energyBend();
    return result;
}

template<typename Real_>
Real_ UmbrellaMesh_T<Real_>::energyTwist() const {
    Real_ result = 0;
    for (const auto &s : m_segments) result += s.rod.energyTwist();
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Derivatives.
////////////////////////////////////////////////////////////////////////////////

// Rod vertex stencil:
//  +---+---+---+---+
//          ^
// Rod edge stencil:
//    +---+---+---+
//          ^
template<typename Real_>
size_t UmbrellaMesh_T<Real_>::hessianNNZ(bool variableDesignParameters) const {
    size_t result = 0;
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = m_segments[si];
        const auto &r = s.rod;
        if (r.numEdges() < 5) throw std::runtime_error("Assumption of at least 5 subdivisions violated."); // verify assumption used to simplify sparsity pattern analysis (fully separated joints)
        // Number of "free" vertices and joints in the rod (independent degrees of freedom that are not controlled by the joints)
        int nfv = int(s.numFreeVertices()), // integers must be signed for formulas below
            nfe = int(s.numFreeEdges());

        result += 6 * nfv + nfe;               // diagonal blocks of x-x and theta-theta terms
        result += 9 * ((nfv - 1) + (nfv - 2)); // contributions from each free vertex to the previous free vertices in the stencil (up to 2 neighbors)

        size_t odiagxt;
        // Add contribution from free thetas to the free vertices in their stencils; depends on number of joints.
        if      (s.numJoints() == 2) { odiagxt = 3 * (2 * 2 + std::min(2, nfe - 2) * std::min(3, nfv) + std::max(nfv - 3, 0) * 4); }
        else if (s.numJoints() == 1) { odiagxt = 3 * (1 * 2 + std::min(2, nfe - 1) * std::min(3, nfv) + std::max(nfv - 3, 0) * 4); }
        else { throw std::runtime_error("Each segment should have exactly two joints"); }
        result += odiagxt;
        result += nfe - 1; // Contributions of thetas to previous thetas in the edge stencil

        if (variableDesignParameters) {
            if (m_umbrella_dPC.restLen && (s.segmentType() == SegmentType::Arm)) {
                // Entries for this segment's rest lengths
                // x-free-restlen interactions are the same as x-theta
                result += odiagxt;
                result += nfe + 2 * (nfe - 1); // theta-restlen block is tri-diagonal (and we take the whole thing)
                result += nfe +      nfe - 1;  // free restlen-restlen part is tridiagonal (and we take only the upper half)

                // restlen-joint blocks: two closest edges on segment X interact with pos, omega, alpha, len_X
                // TODO (Samara): Can this be replaced with s.numJoints() * 2 * (6+2)?
                for (size_t j = 0; j < 2; ++j) {
                    size_t ji = s.joint(j);
                    if (ji == NONE) continue;
                    result += 2 * (6 + 2); // pos, omega, alpha, len_X
                }
            }
        }
    }

    for (const auto &j : m_joints) {
        // All joint variables interact with each other except for the length variables.
        result += 28 + (j.valence() * 8); // upper tri of dense 7x7 block for joint self-interaction, plus the interaction with the length variables but not within the length variable block.

        // The two closest vertices and thetas of all incident segments depend
        // on the joint's position, omega, alpha vars as well as the joint len
        // vars that control only the rod containing that segment
        //                           x   theta     #joint vars
        result += j.valence() * 2 * (3 +   1  ) * (7    +    1);

        if (variableDesignParameters) {            
            // The joint rest lengths interact with the adjacent free vertices/thetas/rest lengths of their corresponding incident segments.
            // We have one of these sets of interactions for each incident segment.
            result += j.numArms() * (3 + 1 + 1);
            // They also interact with the joint variables that affect the corresponding edge vectors
            // ((pos, omega, alpha, len_X) for rod X's rest length
            result += j.numArms() * 8;

            // They also interact with themselves (but not each other).
            result += j.numArms();
        }
    }

    // Additional NNZ contributed by linear actuators. These come from the interations between:
    //      top/bottom positions
    //      top/bottom omegas
    //      top position with bottom omega
    //      bottom position with top omega
    result += numUmbrellas() * (9 * 4);

    result += numRepulsionNNZ();
    return result;
}

template<typename Real_>
auto UmbrellaMesh_T<Real_>::hessianSparsityPattern(bool variableDesignParameters, Real_ val) const -> CSCMat {
    if (variableDesignParameters && m_cachedHessianVarRLSparsity) {
        if (m_cachedHessianVarRLSparsity->Ax[0] != val) m_cachedHessianVarRLSparsity->fill(val);
        return *m_cachedHessianVarRLSparsity;
    }
    if (!variableDesignParameters && m_cachedHessianSparsity) {
        if (m_cachedHessianSparsity->Ax[0] != val) m_cachedHessianSparsity->fill(val);
        return *m_cachedHessianSparsity;
    }

    const size_t nnz = hessianNNZ(variableDesignParameters);
    const size_t ndof = variableDesignParameters ? numExtendedDoF() : numDoF();

    CSCMat result(ndof, ndof);
    result.symmetry_mode = CSCMat::SymmetryMode::UPPER_TRIANGLE;
    result.nz = nnz;
    result.Ap.reserve(ndof + 1);
    result.Ai.reserve(nnz);
    result.Ax.assign(nnz, val);

    auto &Ap = result.Ap;
    auto &Ai = result.Ai;

    // Append the indices [start, end) to Ai
    auto addIdxRange = [&](const size_t start, const size_t end) {
        assert((start <= ndof) && (end <= ndof));
        const size_t len = end - start, oldSize = Ai.size();
        Ai.resize(oldSize + len);
        for (size_t i = 0; i < len; ++i)
            Ai[oldSize + i] = start + i;
    };
    auto addIdx = [&](const size_t idx) { Ai.push_back(idx); };

    auto finalizeCol = [&](bool needsSort = false) {
        const size_t colStart = Ap.back();
        const size_t colEnd = Ai.size();
        Ap.push_back(colEnd);
        if (needsSort)
            std::sort(Ai.begin() + colStart, Ai.begin() + colEnd);
    };

    // Build the sparsity pattern in compressed form one column (variable) at a time.
    result.Ap.push_back(0);
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = m_segments[si];
        const auto &r = s.rod;
        const size_t so = m_dofOffsetForSegment[si];
        if (r.numEdges() < 5) throw std::runtime_error("Assumption of at least 5 subdivisions violated."); // verify assumption used to simplify sparsity pattern analysis (fully separated joints)

        const size_t nfv = s.numFreeVertices(), nfe = s.numFreeEdges();

        // Contribution from free vertices to the earlier free vertices before them in their stencils.
        for (size_t vi = 0; vi < nfv; ++vi) {
            const size_t vstart = vi - std::min<size_t>(2, vi);
            for (size_t c = 0; c < 3; ++c) {
                addIdxRange(so + 3 * vstart, so + 3 * vi + c + 1);
                finalizeCol();
            }
        }

        // Contribution from free thetas to the free vertices, thetas in their stencil
        // We have two indexing cases depending on whether or not there is a
        // start joint (whether the first edge shares the index of the vertex
        // before or after).
        //    +---+)--+---+-...---+--(+---+
        //          0 0 1 1     nfv-1
        //    +---+---+---+-...---+--(+---+
        //    0 0 1 1 2 2 3     nfv-1
        // Edge stencil according to free vertex indexing (when has start joint)
        //  ei-2  ei-1   ei   ei+1
        //    +-----+-----+-----+
        //      ei-1   ^    ei+1
        //            ei
        // Edge stencil according to free vertex indexing (no start joint)
        //  ei-1   ei   ei+1  ei+2
        //    +-----+-----+-----+
        //      ei-1   ^    ei+1
        //            ei
        for (size_t ei = 0; ei < nfe; ++ei) {
            const size_t vstart = ei - std::min<size_t>( s.hasStartJoint() ? 2 : 1, ei);
            const size_t vend   = std::min<size_t>(ei + (s.hasStartJoint() ? 1 : 2) + 1, nfv);
            addIdxRange(so + 3 * vstart, so + 3 * vend);
            const size_t estart = ei - std::min<size_t>(1, ei);
            addIdxRange(so + 3 * nfv + estart, so + 3 * nfv + ei + 1);
            finalizeCol();
        }
    }

    for (size_t ji = 0; ji < numJoints(); ++ji) {
        const auto &j = m_joints[ji];
        const size_t jo = m_dofOffsetForJoint[ji];
        for (size_t d = 0; d < j.numDoF(); ++d) {
            // Contribution from the joint variables to the free vertices and free thetas of the incident segments.
            j.visitInfluencedSegmentVars(d, addIdx);

            // Joint-joint blocks:
            if      (d < j.numBaseDoF()) { addIdxRange(jo, jo + d + 1);             } // (pos, omega, alpha) all interact with each other
            else {
                // The length variable interacts with (pos, omega, alpha) and itself.
                addIdxRange(jo, jo + 7);
                addIdx(jo + d);
            }
            // Deployment energy blocks:
            size_t opposite_ji = j.getOppositeCenter(); 
            if ((opposite_ji != NONE) && (ji > opposite_ji) && d < 6) {
                const size_t opposite_jo = m_dofOffsetForJoint[opposite_ji];
                addIdxRange(opposite_jo, opposite_jo + 6); // interact with the position and omega variables of the opposite center.
            }

            // Replusion energy blocks:
            // Top 
            std::vector<size_t> top_neighbor_ji_vec = top_neighbor_ji(ji); 
            for (size_t nid = 0; nid < top_neighbor_ji_vec.size(); ++nid) {
                size_t n_ji = top_neighbor_ji_vec[nid];
                if ((ji > n_ji) && d < 3) {
                    const size_t n_jo = m_dofOffsetForJoint[n_ji];
                    addIdxRange(n_jo, n_jo + 3); // interact with the position variables of the neighbor center.
                }
            }
            // Bot
            std::vector<size_t> bot_neighbor_ji_vec = bot_neighbor_ji(ji); 
            for (size_t nid = 0; nid < bot_neighbor_ji_vec.size(); ++nid) {
                size_t n_ji = bot_neighbor_ji_vec[nid];
                if ((ji > n_ji) && d < 3) {
                    const size_t n_jo = m_dofOffsetForJoint[n_ji];
                    addIdxRange(n_jo, n_jo + 3); // interact with the position variables of the neighbor center.
                }
            }
            
            finalizeCol(true); // variables weren't added in order; need sorting
        }
    }


    if (variableDesignParameters) {
        if (m_umbrella_dPC.restLen) {
            // Interaction of the segments' rest length variables with the free vertices, thetas, and joint and rest kappa variables.
            for (size_t si = 0; si < numSegments(); ++si) {
                const auto &s = m_segments[si];
                if (s.segmentType() == SegmentType::Arm) {
                    const size_t srlo = m_restLenDofOffsetForSegment[si],
                                 so = m_dofOffsetForSegment[si],
                                 nfv = s.numFreeVertices(), 
                                 nfe = s.numFreeEdges();

                    for (size_t ei = 0; ei < nfe; ++ei) {
                        // Same restlen-vertex interactions as the theta-vertex interactions
                        const size_t vstart = ei - std::min<size_t>( s.hasStartJoint() ? 2 : 1, ei);
                        const size_t vend   = std::min<size_t>(ei + (s.hasStartJoint() ? 1 : 2) + 1, nfv);
                        addIdxRange(so + 3 * vstart, so + 3 * vend);

                        // Rest lengths interact with thetas in the full edge stencil (not just the upper triangle)
                        const size_t estart = ei - std::min<size_t>(1, ei);
                        addIdxRange(so + 3 * nfv + estart, so + 3 * nfv + std::min<size_t>((ei + 1) + 1, nfe));

                        // Rest lengths interact with rest lengths in the upper triangle of the edge stencil
                        addIdxRange(srlo + estart, srlo + ei + 1);

                        // restlen-joint blocks: closest two edges of both segments interact with the position/omega variables.
                        //                       closest two edges of the segment interact with alpha, len
                        auto jointInteraction = [&](const size_t ji) {
                            assert(ji != NONE);
                            const size_t jo = m_dofOffsetForJoint[ji];
                            const auto &  j = m_joints[ji];
                            const size_t len_offset = j.len_offset(si);
                            addIdxRange(jo, jo + 7); // end edge and second-to-end edge interact with position, omega, alpha
                            addIdx(jo + 7 + len_offset); // edges on segment_X interact with len_X,
                        };

                        // Note: these two conditions can overlap when nsubdiv = 5!
                        if ((ei <        2) && s.hasStartJoint()) jointInteraction(s.startJoint);
                        if ((ei >= nfe - 2) && s.  hasEndJoint()) jointInteraction(s.  endJoint);

                        // Joint variable row indices weren't added in order; needs sorting
                        finalizeCol(true);
                    }
                }
            }
            // Interaction of the joints' rest length variables with the free vertices, thetas, and joint variables, free edge rest lengths, and joints rest lengths
            for (size_t ji = 0; ji < numJoints(); ++ji) {
                const auto &j = m_joints[ji];

                for (size_t i = 0; i < j.numArms(); ++i) {
                    // Interactions with all free vertices/thetas/design variables
                    j.visitInfluencedSegmentVars(i, addIdx, true);
                    
                    const size_t jo = m_dofOffsetForJoint[ji];
                    const size_t lsi = j.getSegmentIndexForArmAt(i);
                    // Add interactions with the joint variables
                    addIdxRange(jo, jo + 7); addIdx(jo + 7 + lsi); // (pos, omega, alpha, len_X)

                    // Add the self-interactions of the joint rest lengths
                    addIdx(m_designParameterDoFOffsetForJoint[ji] + i);

                    // Variables weren't added in order; needs sorting
                    finalizeCol(true);
                }
            }
        }
    }

    if (size_t(result.nz) != result.Ai.size()) throw std::runtime_error("Incorrect NNZ prediction: " + std::to_string(result.nz) + " vs " + std::to_string(result.Ai.size()));
    if (variableDesignParameters) { m_cachedHessianVarRLSparsity = std::make_unique<CSCMat>(result); }
    else                 { m_cachedHessianSparsity      = std::make_unique<CSCMat>(result); }

    return result;
}

// Construct sparse (compressed row) representation of dvk_dri; dv_dr[k][i] gives the derivative of
// unconstrained segment variable k with respect to the global reduced
// linkage variables i.
// If segmentJointDofOffset != NONE, the derivatives of unconstrained rest lengths with respect
// to global reduced linkage variables are also computed.
template<typename Real_>
struct dv_dr_entry {
    typename CSCMatrix<SuiteSparse_long, Real_>::index_type first;
    typename CSCMatrix<SuiteSparse_long, Real_>::value_type second;
};
template<typename Real_>
using dv_dr_type = std::vector<std::vector<dv_dr_entry<Real_>>>;


static constexpr size_t JointJacobianRows = 7;
static constexpr size_t JointJacobianCols = 5;

// For correct autodiff code, we must still keep zero entries if they have nonzero derivatives!
bool entryIdenticallyZero(double val) { return val == 0; }
bool entryIdenticallyZero(ADReal val) { return (val == 0) && (val.derivatives().squaredNorm() == 0); }

template<typename Real_, typename LTESPtr>
void
dv_dr_for_segment(const typename UmbrellaMesh_T<Real_>::RodSegment &s,
                  const std::array<LTESPtr, 2> &jointSensitivity,
                  const std::array<size_t, 2> &segmentJointDofOffset,
                  size_t segmentDofOffset,
                  dv_dr_type<Real_> &dv_dr, /* output */
                  // Arguments needed only for variable rest length case
                  const std::array<size_t, 2> &segmentJointRestLenDofOffset = std::array<size_t, 2>(),
                  bool variableDesignParameters = false,
                  bool use_restLen = false,
                  bool /* use_restKappa */ = false,
                  size_t segmentRestLenDofOffset = UmbrellaMesh::NONE,
                  bool skip = false
                  )
{   
    const auto &r = s.rod;
    const size_t nv = r.numVertices(), ne = r.numEdges();
    const size_t numFullDoF = variableDesignParameters ? r.numExtendedDoF() : r.numDoF();

    using index_type = SuiteSparse_long;

    dv_dr.resize(numFullDoF);
    if (skip) return;
    // Joint variables: pos, omega, alpha, len_X
    // pos, omega, alpha, len_X affect segment X terminal vertices (1 pos component + 3 omega components + 1 alpha + 1 len = 6 vars)
    // omega, alpha             affect segment X's terminal theta  (4 vars) (but alpha dependence is 0 if source frame has been updated...)
    for (auto &row : dv_dr) { row.clear(); row.reserve(6); } // At most 6 reduced variables affect each variable.

    for (size_t k = 0; k < numFullDoF; ++k) {
        auto &dvk_dr = dv_dr[k];

        // Derivative of a constrained centerline position
        auto jointVertexSensitivity = [&](bool isTail, index_type comp, size_t localJointIdx) {
            index_type o = index_type(segmentJointDofOffset[localJointIdx]);
            const auto &js = *jointSensitivity[localJointIdx];
            Real_ dx_de = isTail ? -1 : 1;
            dvk_dr.push_back({o + comp, 1.0}); // Derivative with respect to corresponding joint position component
            // Derivatives with respect to the joint orientation, opening angle, and length var
            for (index_type l = 0; l < index_type(JointJacobianCols); ++l) {
                Real_ de_comp_dvar_l = js.jacobian(comp,     l);
                Real_ dp_comp_dvar_l = js.jacobian(comp + 4, l);
                Real_ entry = 0.5 * dx_de * de_comp_dvar_l + dp_comp_dvar_l;
                if (entryIdenticallyZero(entry)) continue;
                // Columns 0..4 of js.jacobian correspond to reduced vars o + 3, ... o + 3 + 3.
                // However the last column (variable l_X) corresponds to o + 3 + 4 + js.localSegmentIndex
                index_type joint_var_index = o + 3 + ((l < 4) ? l : 4 + js.localSegmentIndex);
                dvk_dr.push_back({joint_var_index, entry});
            }
        };

        // Derivative of a constrained edge theta
        auto jointEdgeSensitivity = [&](size_t localJointIdx) {
            index_type o = index_type(segmentJointDofOffset[localJointIdx]);
            const auto &js = *jointSensitivity[localJointIdx];
            // Note: the final l_X variable does not influence "theta"!
            for (index_type l = 0; l < index_type(JointJacobianCols - 1); ++l) {
                Real_ dtheta_dvar_l = js.jacobian(3, l);
                if (entryIdenticallyZero(dtheta_dvar_l)) continue;
                dvk_dr.push_back({o + 3 + l, dtheta_dvar_l});
            }
        };

        if (k < r.thetaOffset()) {
            size_t vtx = k / 3;
            index_type component = k % 3;
            if      ((vtx <       2) && jointSensitivity[0]) jointVertexSensitivity(vtx == 0,      component, 0);
            else if ((vtx >= nv - 2) && jointSensitivity[1]) jointVertexSensitivity(vtx == nv - 2, component, 1);
            else                                             dvk_dr.push_back({index_type(segmentDofOffset + k - 3 * 2 * s.hasStartJoint()), 1.0}); // Permuted Kronecker delta
        }
        else if (k < r.designParameterOffset()) {
            size_t eidx = k - r.thetaOffset();
            if      ((eidx ==      0) && jointSensitivity[0]) jointEdgeSensitivity(0);
            else if ((eidx == ne - 1) && jointSensitivity[1]) jointEdgeSensitivity(1);
            else                                              dvk_dr.push_back({index_type(segmentDofOffset + s.numPosDoF() + eidx - s.hasStartJoint()), 1.0}); // Permuted Kronecker delta
        }
        else {
            assert(variableDesignParameters);
            size_t eidx = k - r.designParameterOffset();
            // The k here is indexing over the dof in a single elastic rod, so we still have rest length before rest kappa. 
            // Rest length variable...
            if (use_restLen && (s.segmentType() ==  UmbrellaMesh_T<Real_>::SegmentType::Arm)) {
                if      ((eidx ==      0) && jointSensitivity[0]) dvk_dr.push_back({index_type(segmentJointRestLenDofOffset[0]), 1.0});
                else if ((eidx == ne - 1) && jointSensitivity[1]) dvk_dr.push_back({index_type(segmentJointRestLenDofOffset[1]), 1.0});
                else if (eidx < ne)                               dvk_dr.push_back({index_type(segmentRestLenDofOffset + eidx - s.hasStartJoint()), 1.0}); // Permuted Kronecker delta
            }
        }
    }
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::hessian(CSCMat &H, UmbrellaEnergyType type, EnergyType eType, const bool variableDesignParameters) const {
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".hessian");

    assert((size_t(H.m) == (variableDesignParameters ? numExtendedDoF() : numDoF())) && (H.n == H.m));

    struct DVDRCustomData : public CustomThreadLocalData {
        dv_dr_type<Real_> dv_dr;
        CSCMat sH; // WARNING: by reusing the per-segment Hessian we assume all rods have the same number of edges.
    };

    // Our Hessian can only be evaluated after the source configuration has
    // been updated; use the more efficient gradient formulas.
    const bool updatedSource = true;
    m_sensitivityCache.update(*this, updatedSource, true /* make sure the joint Hessian is cached */);

    // Assemble the (transformed) Hessian of each rod segment using the
    // gradients of the parameters with respect to the reduced parameters.
    auto assemblePerSegmentHessian = [&](size_t si, CSCMat &Hout, DVDRCustomData &customData) {

        // BENCHMARK_START_TIMER_SECTION("Segment hessian preamble");
        const auto &s = m_segments[si];
        const auto &r = s.rod;
        auto &dv_dr = customData.dv_dr;

        // BENCHMARK_START_TIMER_SECTION("Rod hessian + grad");
        // Gradient and Hessian with respect to the segment's unconstrained DoFs
        auto &sH = customData.sH;
        if (sH.m == 0) sH = r.hessianSparsityPattern(variableDesignParameters);
        else           sH.template setZero</*multithreaded = */ false>();

        r.hessian(sH, eType, variableDesignParameters);
        const auto sg = r.template gradient<GradientStencilMaskTerminalsOnly>(updatedSource, eType); // we never need the variable rest length gradient since the mapping from global to local rest lengths is linear
        // BENCHMARK_STOP_TIMER_SECTION("Rod hessian + grad");

        size_t segmentDofOffset = m_dofOffsetForSegment[si];
        // BENCHMARK_STOP_TIMER_SECTION("Segment hessian preamble");

        // Sensitivity of terminal edges to the start/end joints (if they exist)
        // BENCHMARK_START_TIMER_SECTION("UmbrellaMeshTerminalEdgeSensitivity");
        std::array<const UmbrellaMeshTerminalEdgeSensitivity<Real_> *, 2> jointSensitivity{{ nullptr, nullptr }};
        std::array<size_t, 2> segmentJointDofOffset, segmentJointRestLenDofOffset;
        for (size_t i = 0; i < 2; ++i) {
            size_t ji = s.joint(i);
            if (ji == NONE) continue;
            jointSensitivity[i] = &m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            segmentJointDofOffset[i] = m_dofOffsetForJoint[ji];

            if (variableDesignParameters && m_umbrella_dPC.restLen) {
                // Index of rest global length variable controlling segment si's end at local joint i
                size_t jointRestLengthOffset = m_joints[ji].arm_offset_for_global_segment(si);
                segmentJointRestLenDofOffset[i] = m_designParameterDoFOffsetForJoint[ji] + jointRestLengthOffset;
            }
        }
        // BENCHMARK_STOP_TIMER_SECTION("UmbrellaMeshTerminalEdgeSensitivity");

        // BENCHMARK_START_TIMER_SECTION("dv_dr_for_segment");

        dv_dr_for_segment(s, jointSensitivity, segmentJointDofOffset, segmentDofOffset, dv_dr, segmentJointRestLenDofOffset, variableDesignParameters, m_umbrella_dPC.restLen, m_umbrella_dPC.restKappa, m_restLenDofOffsetForSegment[si]);

        // BENCHMARK_STOP_TIMER_SECTION("dv_dr_for_segment");

        // BENCHMARK_START_TIMER_SECTION("rod hessian contrib");
        // Accumulate contribution of each (upper triangle) entry in sH to the
        // full Hessian term:
        //      dvk_dri sH_kl dvl_drj
        // This step still takes a majority of the time despite optimization efforts...
        // Entries in dv_dr tend to be contiguous, so we only use a binary search to find
        // the first output entry for a given column.
        using Idx = typename CSCMat::index_type;
        Idx idx = 0, idx2 = 0;
        Idx ncol = sH.n, colbegin = sH.Ap[0];
        // For each active hessian entry in the segment hessian, we loop through the reduced variables with which vk and vl are coupled.
        for (Idx l = 0; l < ncol; ++l) {
            const Idx colend = sH.Ap[l + 1];
            for (auto entry = colbegin; entry < colend; ++entry) {
                const Idx k = sH.Ai[entry];
                const auto v = sH.Ax[entry];
                assert(k <= l);
                const auto &dvk_dr = dv_dr[k];
                const auto &dvl_dr = dv_dr[l];
                for (const auto &dvl_drj : dvl_dr) {
                    const Idx j = dvl_drj.first;
                    {
                        if (dvk_dr.size() == 0) continue;
                        const Idx i = dvk_dr[0].first;
                        if (i > j) continue;
                        if ((idx >= Hout.Ap[j + 1]) || (Hout.Ai[idx] != i) || (idx < Hout.Ap[j])) {
                            idx = Hout.findEntry(i, j);
                        }
                    }
                    const auto val = dvl_drj.second * v;
                    Hout.Ax[idx++] += val * dvk_dr[0].second;
                    for (size_t ii = 1; ii < dvk_dr.size(); ++ii) {
                        const Idx i = dvk_dr[ii].first;
                        if (i > j) break;
                        while (Hout.Ai[idx] < i) ++idx;
                        Hout.Ax[idx++] += val * dvk_dr[ii].second;
                    }
                }
                if (k != l) {
                    // Contribution from (l, k), if it falls in the upper triangle of H; capture all the missed entries from the previous loop due to the (i>j) check.
                    for (const auto &dvk_drj : dvk_dr) {
                        const Idx j = dvk_drj.first;
                        {
                            if (dvl_dr.size() == 0) continue;
                            const Idx i = dvl_dr[0].first;
                            if (i > j) continue;
                            if ((idx2 >= Hout.Ap[j + 1]) || (Hout.Ai[idx2] != i) || (idx2 < Hout.Ap[j])) {
                                idx2 = Hout.findEntry(i, j);
                            }
                        }
                        const auto val = dvk_drj.second * v;
                        Hout.Ax[idx2++] += val * dvl_dr[0].second;
                        for (size_t ii = 1; ii < dvl_dr.size(); ++ii) {
                            const Idx i = dvl_dr[ii].first;
                            if (i > j) break;
                            while (Hout.Ai[idx2] < i) ++idx2;
                            Hout.Ax[idx2++] += val * dvl_dr[ii].second;
                        }
                    }
                }
            }
            colbegin = colend;
        }

        // BENCHMARK_STOP_TIMER_SECTION("rod hessian contrib");

        // BENCHMARK_START_TIMER_SECTION("joint hessian contrib");
        // Accumulate contribution of the Hessian of e^j and theta^j wrt the joint parameters.
        //      dE/var^j (d^2 var^j / djoint_var_k djoint_var_l)
        for (size_t ji = 0; ji < 2; ++ji) {
            if (jointSensitivity[ji] == nullptr) continue;
            const auto &js = *jointSensitivity[ji];
            const size_t o = segmentJointDofOffset[ji] + 3; // DoF index for first component of omega
            Vec3 dE_de_j = 0.5 * (sg.gradPos(js.j + 1) - sg.gradPos(js.j));
            Vec3 dE_dp_j = sg.gradPos(js.j + 1) + sg.gradPos(js.j);
            Real_ dE_dtheta_j = sg.gradTheta(js.j);
            Eigen::Matrix<Real_, JointJacobianCols, JointJacobianCols> contrib;
            contrib =  dE_de_j[0] * js.hessian[0]
                    +  dE_de_j[1] * js.hessian[1]
                    +  dE_de_j[2] * js.hessian[2]
                    + dE_dtheta_j * js.hessian[3]
                    +  dE_dp_j[0] * js.hessian[4]
                    +  dE_dp_j[1] * js.hessian[5]
                    +  dE_dp_j[2] * js.hessian[6];
            for (size_t l = 0; l < JointJacobianCols; ++l) {
                Hout.addNZStrip(o, (l < 4) ? (o + l) : (o + 4 + js.localSegmentIndex),
                                contrib.col(l).topRows(std::min<size_t>(l + 1, 4)));
            }
            Hout.addDiagEntry(o + 4 + js.localSegmentIndex, contrib(4, 4));
        }
        // BENCHMARK_STOP_TIMER_SECTION("joint hessian contrib");

        // BENCHMARK_STOP_TIMER("Accumulate Contributions");
    };

    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::Elastic) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("elastic");
#if 1
        assemble_parallel<DVDRCustomData>(assemblePerSegmentHessian, H, numSegments());
#else
     DVDRCustomData customData;
     for (size_t si = 0; si < numSegments(); ++si) assemblePerSegmentHessian(si, H, customData);
#endif
    }
    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::       Deployment)   addDeploymentHessian(H);
    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::        Repulsion)    addRepulsionHessian(H);
    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::       Attraction)   addAttractionHessian(H);
    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::AngleBoundPenalty) addAnglePenaltyHessian(H);
}

template<typename Real_>
auto UmbrellaMesh_T<Real_>::hessian(UmbrellaEnergyType type, EnergyType eType, bool variableDesignParameters) const -> TMatrix {
    auto H = hessianSparsityPattern(variableDesignParameters);
    hessian(H, type, eType, variableDesignParameters);
    return H.getTripletMatrix();
}


template<typename Real_>
void UmbrellaMesh_T<Real_>::massMatrix(CSCMat &M, bool updatedSource, bool useLumped) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".massMatrix");
    assert(M.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);

    {
        // Also cache the joint parametrization Hessian if it can be computed accurately
        // (if the source frames are up-to-date); we should almost always want the Hessian
        // too when asking for the Mass matrix.
        const bool evalHessian = updatedSource;
        m_sensitivityCache.update(*this, updatedSource, evalHessian);
    }

    struct DVDRCustomData : public CustomThreadLocalData {
        dv_dr_type<Real_> dv_dr;
    };

    // Assemble the (transformed) mass matrix of each rod segment using the
    // gradients of the parameters with respect to the reduced parameters.
    auto assemblePerSegmentMassMatrix = [&](size_t si, CSCMat &Mout, DVDRCustomData &customData) {
        const auto &s = m_segments[si];
        const auto &r = s.rod;
        dv_dr_type<Real_> &dv_dr = customData.dv_dr;
        size_t segmentDofOffset = m_dofOffsetForSegment[si];

        std::array<const UmbrellaMeshTerminalEdgeSensitivity<Real_> *, 2> jointSensitivity{{ nullptr, nullptr }};
        std::array<size_t, 2> segmentJointDofOffset;
        for (size_t i = 0; i < 2; ++i) {
            size_t ji = s.joint(i);
            if (ji == NONE) continue;
            jointSensitivity[i] = &m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            segmentJointDofOffset[i] = m_dofOffsetForJoint[ji];
        }

        dv_dr_for_segment(s, jointSensitivity, segmentJointDofOffset, segmentDofOffset, dv_dr);

        // Mass matrix with respect to the segment's unconstrained DoFs
        CSCMat sM;
        if (useLumped) { sM.setDiag(r.lumpedMassMatrix()); }
        else           { sM = r.hessianSparsityPattern();
                              r.massMatrix(sM); }

        // Accumulate contribution of each (upper triangle) entry in sM to the
        // full mass matrix term:
        //      dvk_dri M_kl dvl_drj
        size_t hint = 0;
        for (const auto t : sM) {
            const size_t k = t.i, l = t.j;
            assert(k <= l);
            for (const auto &dvl_drj : dv_dr.at(l)) {
                size_t j = dvl_drj.first;
                for (const auto &dvk_dri : dv_dr.at(k)) {
                    size_t i = dvk_dri.first;
                    Real_ val = dvk_dri.second * dvl_drj.second * t.v;
                    // Accumulate contributions from sM's upper triangle entry
                    // (k, l) and, if in the strict upper triangle, its
                    // corresponding lower triangle entry (l, k). This
                    // corresponding entry is found by exchanging i, j.
                    // Of course, we only keep contributions in the upper
                    // triangle of M.
                    if ( i <= j             ) hint = Mout.addNZ(i, j, val, hint); // contribution from (k, l), if it falls in the upper triangle of H.
                    if ((j <= i) && (k != l)) hint = Mout.addNZ(j, i, val, hint); // contribution from (l, k), if it falls in the upper triangle of H and wasn't already added.
                }
            }
        }
    };

#if 0
    assemble_parallel<DVDRCustomData>(assemblePerSegmentMassMatrix, M, numSegments());
#else
    DVDRCustomData cdata;
    for (size_t si = 0; si < numSegments(); ++si) assemblePerSegmentMassMatrix(si, M, cdata);
#endif
}

// Diagonal lumped mass matrix constructed by summing the mass in each row.
// WARNING: this matrix is usually not positive definite!
template<typename Real_>
auto UmbrellaMesh_T<Real_>::lumpedMassMatrix(bool /* updatedSource */) const -> VecX {
    const size_t ndof = numDoF();
#if 0
    auto M = massMatrix(updatedSource, true);
    Eigen::VectorXd Mdiag = Eigen::VectorXd::Zero(ndof);

    for (const auto &t : M) {
        Mdiag[t.i] += t.v;
        if (t.j != t.i) Mdiag[t.j] += t.v;
    }

    if (Mdiag.minCoeff() <= 0) throw std::runtime_error("Lumped mass matrix is non-positive");

    return Mdiag;
#else
    return VecX::Ones(ndof);
#endif
}

template<typename Real_>
Real_ UmbrellaMesh_T<Real_>::approxLinfVelocity(const VecX &paramVelocity) const {
    const size_t ndof = numDoF();
    TMatrix M(ndof, ndof);
    M.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;

    if ((size_t)paramVelocity.size() != ndof) throw std::runtime_error("The input argument for approxLinfVelocity doesn't match the degrees of freedom of the system!");
    const bool updatedSource = true; // The elastic rod approxLinfVelocity formulas already assume an updated source frame...
    {
        // We should almost always want the Hessian too when asking for the
        // Mass matrix...
        const bool evalHessian = updatedSource;
        m_sensitivityCache.update(*this, updatedSource, evalHessian);
    }

    VecX rodParamVelocity;

    // Assemble the (transformed) mass matrix of each rod segment using the
    // gradients of the parameters with respect to the reduced parameters.
    Real_ maxvel = 0;
    for (size_t si = 0; si < numSegments(); ++si) {
        const auto &s = m_segments[si];
        const auto &r = s.rod;
        const size_t nv = r.numVertices(), ne = r.numEdges();

        // Apply chain rule to determine the velocity of the rod's unconstrained DoFs
        rodParamVelocity.resize(r.numDoF());
        rodParamVelocity.setZero();

        // Copy over the velocities for the degrees of freedom that
        // directly control the interior/free-end centerline positions and
        // material frame angles.
        size_t offset = m_dofOffsetForSegment[si];
        for (size_t i = 0; i < nv; ++i) {
            // The first/last edge don't contribute degrees of freedom if they're part of a joint.
            if ((i <       2) && s.hasStartJoint()) continue;
            if ((i >= nv - 2) && s.  hasEndJoint()) continue;
            rodParamVelocity.template segment<3>(3 * i) = paramVelocity.template segment<3>(offset);
            offset += 3;
        }
        for (size_t j = 0; j < ne; ++j) {
            if ((j ==      0) && s.hasStartJoint()) continue;
            if ((j == ne - 1) && s.  hasEndJoint()) continue;
            rodParamVelocity[3 * nv + j] = paramVelocity[offset++];
        }

        // Set velocities induced by the start/end joints (if they exist)
        for (size_t i = 0; i < 2; ++i) {
            size_t jindex = s.joint(i);
            if (jindex == NONE) continue;
            const size_t offset = m_dofOffsetForJoint.at(jindex);
            const auto &sensitivity = m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            const size_t j = sensitivity.j;
            //           pos     e_X    theta^j
            // x_j     [  I    -0.5 I     0   ] [ I 0 ... 0]
            // x_{j+1} [  I     0.5 I     0   ] [ jacobian ]
            // theta^j [  0         0     I   ]
            Vec3 d_edge = sensitivity.jacobian.template block<3, JointJacobianCols>(0, 0) * paramVelocity.template segment<JointJacobianCols>(offset + 3);
            rodParamVelocity.template segment<3>(3 * (j    )) = paramVelocity.template segment<3>(offset) - 0.5 * d_edge;
            rodParamVelocity.template segment<3>(3 * (j + 1)) = paramVelocity.template segment<3>(offset) + 0.5 * d_edge;
            rodParamVelocity                    [3 * nv + j]  = sensitivity.jacobian.template block<1, JointJacobianCols>(3, 0) * paramVelocity.template segment<JointJacobianCols>(offset + 3);
        }

        maxvel = std::max(maxvel, r.approxLinfVelocity(rodParamVelocity));
    }

    return maxvel;
}

template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::getExtendedDoFsPARL() const {
    VecX result(numExtendedDoFPARL());
    const size_t nrs = numArmSegments();
    result.head(numDoF()) = getDoFs();
    if (m_umbrella_dPC.restLen) result.tail(nrs) = m_perArmRestLen;
    return result;
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::setExtendedDoFsPARL(const VecX &params, bool spatialCoherence) {
    if (size_t(params.size()) != numExtendedDoFPARL()) throw std::runtime_error("Extended DoF size mismatch");
    setDesignParameters(params.tail(numDesignParams()));
    setDoFs(params.head(numDoF()), spatialCoherence);
}

template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::gradientPerArmRestlen(bool updatedSource, UmbrellaEnergyType type, EnergyType eType) const {
    auto gPerEdgeRestLen = gradient(updatedSource, type, eType, true);
    VecX result(numExtendedDoFPARL());
    result.head(numDoF()) = gPerEdgeRestLen.head(numDoF());
    if (m_umbrella_dPC.restLen) m_armRestLenToEdgeRestLenMapTranspose.applyRaw(gPerEdgeRestLen.tail(numRestLengths()).data(), result.tail(numArmSegments()).data(), /* no transpose */ false);
    return result;
}

template<typename Real_>
auto UmbrellaMesh_T<Real_>::hessianPerArmRestlenSparsityPattern(Real_ val) const -> CSCMat {
    if (m_cachedHessianPARLSparsity) return *m_cachedHessianPARLSparsity;

    auto hspPerEdge = hessianSparsityPattern(true, 0.0);

    const size_t restLenOffset = numDoF();
    const size_t nedparl = restLenOffset + numArmSegments() * m_umbrella_dPC.restLen;
    TMatrix result(nedparl, nedparl);
    result.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;

    auto isRestLen = [&](const size_t i) { return i >= restLenOffset; };

    const SuiteSparseMatrix &S = m_armRestLenToEdgeRestLenMapTranspose;
    for (const auto t : hspPerEdge) {
        // i <= j, so "i" can only be a rest length if "j" is as well
        if (isRestLen(t.j)) {
            if (m_umbrella_dPC.restLen) {
                // Loop over the segments affecting rest length "j"
                const size_t sj = t.j - restLenOffset;
                const size_t ibegin = S.Ap.at(sj), iend = S.Ap.at(sj + 1);
                // Apply the S matrix if it is restlen-restlen terms, otherwise accumulate the cross terms for edges to cross terms for segments. 
                for (size_t idx = ibegin; idx < iend; ++idx) {
                    size_t j = S.Ai[idx] + restLenOffset;

                    if (isRestLen(t.i)) {
                        // Loop over the segments affecting rest length "i"
                        const size_t si = t.i - restLenOffset;
                        const size_t ibegin2 = S.Ap.at(si), iend2 = S.Ap.at(si + 1);
                        for (size_t idx2 = ibegin2; idx2 < iend2; ++idx2) {
                            size_t i = S.Ai[idx2] + restLenOffset;
                            if (i <= j) result.addNZ(i, j, 1.0);
                            if ((t.i != t.j) && (j <= i)) result.addNZ(j, i, 1.0);
                        }
                    }
                    else {
                        // Guaranteed t.i < j
                        result.addNZ(t.i, j, 1.0);
                    }

                }
            }
        }
        else {
            // Identity block
            result.addNZ(t.i, t.j, 1.0);
        }
    }
    m_cachedHessianPARLSparsity = std::make_unique<CSCMat>(result);
    m_cachedHessianPARLSparsity->fill(val);
    return *m_cachedHessianPARLSparsity;
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::hessianPerArmRestlen(CSCMat &H, UmbrellaEnergyType type, EnergyType eType) const {
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    BENCHMARK_SCOPED_TIMER_SECTION timer("hessianPerArmRestlen");
    const size_t restLenOffset = numDoF();
    const size_t ndof = restLenOffset + numArmSegments() * m_umbrella_dPC.restLen;
    assert((size_t(H.m) == ndof) && (size_t(H.n) == ndof));
    UNUSED(ndof);

    // Compute Hessian using per-edge rest lengths
    auto HPerEdge = hessianSparsityPattern(true);
    hessian(HPerEdge, type, eType, true);

    // Leverage the fact that HPerEdge is upper triangular: i <= j.
    // Copy the deformed configuration, rest kappa entries over (identity block)
    // Note: H may have additional entries in its sparsity pattern since, e.g.,
    // the caller may have added regularization terms.
    H.addWithSubSparsity(HPerEdge, /* scale */ 1.0, /*  idx offset */ 0, /* block start */ 0, /* block end */ restLenOffset);

    if (m_umbrella_dPC.restLen) {
        // Use m_armRestLenToEdgeRestLenMapTranspose to transform the rest
        // length part of the Hessian.
        const SuiteSparseMatrix &S = m_armRestLenToEdgeRestLenMapTranspose;
        const size_t nrl = numRestLengths();
        size_t hint = 0;
        for (size_t rlj = 0; rlj < nrl; ++rlj) {
            const size_t j = rlj + restLenOffset;
            // Loop over each output column "l" generated by per-edge rest length "j"
            const size_t lend = S.Ap[rlj + 1];
            for (size_t idx = S.Ap[rlj]; idx < lend; ++idx) {
                const size_t l = S.Ai[idx] + restLenOffset;
                const Real_ colMultiplier = S.Ax[idx];

                // Create entries for each input Hessian entry
                const size_t input_end = HPerEdge.Ap[j + 1];
                for (size_t idx_in = HPerEdge.Ap[j]; idx_in < input_end; ++idx_in) {
                    const Real_ colVal = colMultiplier * HPerEdge.Ax[idx_in];
                    const size_t i = HPerEdge.Ai[idx_in];
                    if (i < restLenOffset) { // left transformation is in the identity block
                        hint = H.addNZ(i, l, colVal, hint);
                    }
                    else {
                        // Loop over each output entry
                        const size_t rli = i - restLenOffset;
                        size_t kprev = 0;
                        size_t kprev_idx = 0;
                        const size_t outrow_end = S.Ap[rli + 1];
                        for (size_t outrow_idx = S.Ap[rli]; outrow_idx < outrow_end; ++outrow_idx) {
                            const size_t k = S.Ai[outrow_idx] + restLenOffset;
                            const Real_ val = S.Ax[outrow_idx] * colVal;
                            if (k <= l) {
                                // Accumulate entries from input's upper triangle
                                if (k == kprev) { H.addNZ(kprev_idx, val); }
                                else     { hint = H.addNZ(k, l, val, hint);
                                           kprev = k, kprev_idx = hint - 1; }
                            }
                            if ((i != j) && (l <= k)) H.addNZ(l, k, val); // accumulate entries from input's (strict) lower triangle
                        }
                    }
                }
            }
        }
    }
}

template<typename Real_>
auto UmbrellaMesh_T<Real_>::hessianPerArmRestlen(UmbrellaEnergyType type, EnergyType eType) const -> TMatrix {
    auto H = hessianPerArmRestlenSparsityPattern();
    hessianPerArmRestlen(H, type, eType);
    return H.getTripletMatrix();
}

////////////////////////////////////
// Forces
////////////////////////////////////
template<typename Real_>
typename UmbrellaMesh_T<Real_>::VecX UmbrellaMesh_T<Real_>::rivetForces(UmbrellaEnergyType type,EnergyType eType, bool needTorque) const {
    if (needTorque) {
        for (const auto &j : m_joints)
            if (j.omega().norm() != 0.0) throw std::runtime_error("Please update the rotation parametrization (updateRotationParametrizations()) for physically meaningful torques.");
    }
    return -gradient(true, type, eType, /* variableDesignParameters */ false, /* designParameterOnly */ false);
}


template<typename Real_>
Eigen::MatrixXd UmbrellaMesh_T<Real_>::UmbrellaRivetNetForceAndTorques(UmbrellaEnergyType type,EnergyType eType) const {
    auto rf = rivetForces(type, eType);
    const size_t nu = numUmbrellas();
    Eigen::MatrixXd result(nu * 2, 6);

    VectorField<double, 3> jointForce, jointTorque;
    for (size_t ui = 0; ui < nu; ++ui) {
        std::array<size_t, 2> uji{{ getUmbrellaCenterJi(ui, 0), getUmbrellaCenterJi(ui, 1)}};
        for (size_t i = 0; i < 2; ++ i) {
            result.block<1, 6>(ui * 2 + i, 0) = stripAutoDiff(rf.template segment<6>(m_dofOffsetForJoint[uji[i]]).eval());
        }
    }
    return result;
}
////////////////////////////////////////////////////////////////////////////////
// Joint operations
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
UmbrellaMesh_T<Real_>::Joint::Joint(UmbrellaMesh_T *um, const Pt3 &p, const Real_ alpha, const Vec3 &normal, const Vec3 &bisector,
                                    const std::vector<TerminalEdgeInputData> &inputTerminalEdges,
                                    const JointType &jointType, const std::vector<size_t> &umbrella_ID)
    : m_umbrella_mesh(um), m_pos(p), m_alpha(alpha), 
      m_ghost_source_t(bisector), m_ghost_source_normal(normal),
      m_jointType(jointType), m_umbrella_ID(umbrella_ID)
{
    num_A_segments = num_B_segments = 0;
    for (const auto &te : inputTerminalEdges) {
        if (te.is_A) ++num_A_segments;
        else         ++num_B_segments;
    }

    const size_t num_segments = num_A_segments + num_B_segments;

    // Transformation from the world basis to the joint frame basis.
    Mat3 joint_source_frame_inv;
    joint_source_frame_inv << m_ghost_source_t.transpose(),
                            ghost_source_nxb().transpose(),
                         m_ghost_source_normal.transpose();
    if ((joint_source_frame_inv.transpose() * joint_source_frame_inv - Mat3::Identity()).norm() > 1e-8)
        throw std::runtime_error("Joint source frame is not orthonormal");

    m_segments.resize(num_segments); m_len.resize(num_segments); m_isStart.resize(num_segments); m_input_t.resize(3, num_segments); m_input_normal.resize(3, num_segments); m_input_p.resize(3, num_segments); 

    int back = 0;
    // TODO: Update joint constructor to use an IO object. In that case we might not need the following assumption.
    // Add A segments first, then B segments. 
    for (int AB = 1; AB >= 0; --AB) {
        for (const auto &te : inputTerminalEdges) {
            if (te.is_A == AB) {
                m_segments[back]         = te.si;
                m_len[back]              = te.len;
                m_isStart[back]          = te.isStart;
                m_input_t     .col(back) = joint_source_frame_inv * te.world_t;
                m_input_normal.col(back) = joint_source_frame_inv * te.world_normal;
                m_input_p     .col(back) = joint_source_frame_inv * te.world_p;
                ++back;
            }
        }
    }

    if (num_A_segments == 0 || num_B_segments == 0) {
        if (m_jointType != JointType::Rigid)
            std::cout << "WARNING: degenerate nonrigid joint set to rigid." << std::endl;
        m_jointType = JointType::Rigid;
    }

    m_armIndexForSegment.assign(num_segments, size_t(NONE));
    m_segmentIndexForArm.clear();
    for (size_t idx  = 0; idx < num_segments; ++idx) {
        auto stype = um->segment(m_segments[idx]).segmentType();
        
        size_t neighbor_idx, curr_idx;
        if (m_isStart[idx]) curr_idx = um->segment(m_segments[idx]).startJoint, neighbor_idx = um->segment(m_segments[idx]).endJoint;
        else curr_idx = um->segment(m_segments[idx]).endJoint, neighbor_idx = um->segment(m_segments[idx]).startJoint;
        
        if(m_jointType == JointType::X) m_jointPosType = JointPosType::Arm;
        else if(m_jointType == JointType::Rigid) {
            // Rigid joint can be plate center or boundary plate edge. For boundary plate edge, the next two lines would do nothing.
            if (um->m_umbrella_to_top_bottom_joint_map[m_umbrella_ID[0]][0] == curr_idx) m_jointPosType = JointPosType::Top;
            else if (um->m_umbrella_to_top_bottom_joint_map[m_umbrella_ID[0]][1] == curr_idx) m_jointPosType = JointPosType::Bot;
            else { // Boundary plate edge
                if (um->m_umbrella_to_top_bottom_joint_map[m_umbrella_ID[0]][0] == neighbor_idx) m_jointPosType = JointPosType::Top;
                else if (um->m_umbrella_to_top_bottom_joint_map[m_umbrella_ID[0]][1] == neighbor_idx) m_jointPosType = JointPosType::Bot;
            }
        }
        else if(m_jointType == JointType::T) {
            if(stype == SegmentType::Plate) {
                if (um->m_umbrella_to_top_bottom_joint_map[m_umbrella_ID[0]][0] == neighbor_idx) m_jointPosType = JointPosType::Top;
                else if (um->m_umbrella_to_top_bottom_joint_map[m_umbrella_ID[0]][1] == neighbor_idx) m_jointPosType = JointPosType::Bot;
                else throw std::runtime_error("T joint must have a neighbor in the top_bottom_joint_map for a segment of type Plate");
            }
        }
        
        if (stype == SegmentType::Arm) {
            m_armIndexForSegment[idx] = m_segmentIndexForArm.size();
            m_segmentIndexForArm.push_back(idx);
        }
    }
    
    m_numArmSegments = m_segmentIndexForArm.size();

    m_omega.setZero();
    m_update();
}

template<typename Real_>
template<class Derived>
void UmbrellaMesh_T<Real_>::Joint::setParameters(const Eigen::DenseBase<Derived> &vars) {
    if (size_t(vars.size()) < numDoF()) throw std::runtime_error("DoF indices out of bounds");
    // parameters: position, omega, alpha, list of len of A edges, list of len of B edges.
    m_pos   = vars.template segment<3>(0);
    m_omega = vars.template segment<3>(3);
    m_alpha = vars[6];
    m_len = vars.segment(7, num_A_segments + num_B_segments);
    m_update();
}

template<typename Real_>
template<class Derived>
void UmbrellaMesh_T<Real_>::Joint::getParameters(Eigen::DenseBase<Derived> &vars) const {
    if (size_t(vars.size()) < numDoF()) throw std::runtime_error("DoF indices out of bounds");
    // 9 parameters: position, omega, alpha, len a, len b
    vars.template segment<3>(0) = m_pos;
    vars.template segment<3>(3) = m_omega;
    vars[6]                     = m_alpha;
    vars.segment(7, valence())  = m_len;
}

// Update the network's full collection of rod points and twist angles with the
// values determined by this joint's configuration (editing only the values
// related to the incident terminal rod edges).
// The "rodSegments" are needed to compute material frame angles from the
// material axis vector.
// "spatialCoherence" determines whether the 2Pi offset ambiguity in theta is
// resolved by minimizing twisting energy (true) or minimizing the change made (temporal coherence; false)
template<typename Real_>
void UmbrellaMesh_T<Real_>::Joint::applyConfiguration(const std::vector<RodSegment>   &rodSegments,
                                                    std::vector<std::vector<Pt3>>   &networkPoints,
                                                    std::vector<std::vector<Real_>> &networkThetas,
                                                    bool spatialCoherence) const {
    // Vector "e" always points outward from the joint into/along the rod.
    auto configureEdge = [&](Vec3 e, bool isStart, const ElasticRod_T<Real_> &rod, std::vector<Pt3> &pts, std::vector<Real_> &thetas, Vec3 normal, int normal_sign, Vec3 offset) {
        const size_t nv = pts.size();
        const size_t ne = thetas.size();
        assert(nv == rod.numVertices());
        assert(ne == rod.numEdges());

        const size_t edgeIdx = isStart ? 0 : ne - 1;

        pts[isStart ? 0 : nv - 2] = m_pos + offset - 0.5 * e;
        pts[isStart ? 1 : nv - 1] = m_pos + offset + 0.5 * e;

        // Material axis d2 is given by the normal vector.

        thetas[edgeIdx] = rod.thetaForMaterialFrameD2(normal_sign * normal, e, edgeIdx, spatialCoherence);
    };

    // Configure the segments attached to ghost edge A and B. 
    for (size_t si = 0; si < num_A_segments + num_B_segments; ++ si) {
        configureEdge(m_e.col(si), m_isStart[si], rodSegments[m_segments[si]].rod, networkPoints[m_segments[si]], networkThetas[m_segments[si]], m_segment_normal.col(si), /* normalSign */ 1, m_p.col(si));

    }
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::Joint::m_update_source_info() {
    m_source_t     .resize(3, valence());
    m_source_normal.resize(3, valence());
    m_source_p     .resize(3, valence());

    if (num_A_segments > 0) {
        Mat3 ghost_source_config_A = get_ghost_source_frame_A();

        m_source_t.     leftCols(num_A_segments) = ghost_source_config_A * m_input_t.     leftCols(num_A_segments);
        m_source_normal.leftCols(num_A_segments) = ghost_source_config_A * m_input_normal.leftCols(num_A_segments);
        m_source_p.     leftCols(num_A_segments) = ghost_source_config_A * m_input_p.     leftCols(num_A_segments);
    }
    if (num_B_segments > 0) {
        Mat3 ghost_source_config_B = get_ghost_source_frame_B();

        m_source_t     .rightCols(num_B_segments) = ghost_source_config_B * m_input_t     .rightCols(num_B_segments);
        m_source_normal.rightCols(num_B_segments) = ghost_source_config_B * m_input_normal.rightCols(num_B_segments);
        m_source_p     .rightCols(num_B_segments) = ghost_source_config_B * m_input_p     .rightCols(num_B_segments);
    }
}

// Update cache; to be called whenever the edge vectors change.
template<typename Real_>
void UmbrellaMesh_T<Real_>::Joint::m_update() {
    m_update_source_info();

    if (num_A_segments > 0) { m_ghost_A = ropt::rotated_matrix(m_omega, ghost_source_t_A()); }
    if (num_B_segments > 0) { m_ghost_B = ropt::rotated_matrix(m_omega, ghost_source_t_B()); }
    m_e = ropt::rotated_matrix(m_omega, m_source_t);
    // Multiply each col with the corresponding length value.
    //    To do that we need a row wise broadcast.
    m_e = m_e.array().rowwise() * m_len.transpose().array();
    m_segment_normal = ropt::rotated_matrix(m_omega, m_source_normal);
    m_p = ropt::rotated_matrix(m_omega, m_source_p);
}

template<typename Real_>
std::tuple<bool, bool, size_t> UmbrellaMesh_T<Real_>::Joint::terminalEdgeIdentification(size_t si) const {
    size_t localSegIndex = 0;
    bool isStart = false;
    bool found_segment = false;
    bool isA = true;

    for (size_t idx = 0; idx < num_A_segments + num_B_segments; ++idx) {
        if (m_segments[idx] == si) {
            localSegIndex = idx;
            isStart = m_isStart[idx];
            found_segment = true;
            isA = idx < num_A_segments;
            break;
        }
    } 
    if (!found_segment) throw std::runtime_error("The segment is not connected to the joint! No terminal edge information!");

    return std::make_tuple(isA, isStart, localSegIndex);
}

// Call "visitor(idx)" for each global independent vertex/theta of
// freedom index "idx" influenced by the joint's variable "var"
// (i.e. that appears in the joint var's column of the Hessian).
// Note: global degrees of freedom are *not* visited in order.
// restLenVar: whether "var" selects an ordinary joint variable or
// a joint rest length variable.
template<typename Real_>
template<class F>
void UmbrellaMesh_T<Real_>::Joint::visitInfluencedSegmentVars(const size_t joint_var_ind, F &visitor, bool restLenVar) const {
    assert(m_umbrella_mesh);
    const auto &um = *m_umbrella_mesh;
    // Visit the affected variables of segment "si".
    // closestOnly: whether to visit only the closest free vertex/theta instead of the closest two.
    auto visitSegment = [&](const size_t si, const bool isStart, bool closestOnly) {
        if (si == NONE) return;
        const size_t so = um.dofOffsetForSegment(si);
        const auto &s = um.segment(si);
        const size_t nfv = s.numFreeVertices(), nfe = s.numFreeEdges();

        // Intervals [vstart, vend), [estart, eend) of affected entities.
        size_t vstart, vend, estart, eend;
        if (isStart) {
            vstart = 0, vend = closestOnly ? 1 : 2;
            estart = 0, eend = closestOnly ? 1 : 2;
        }
        else {
            vstart = closestOnly ? nfv - 1 : nfv - 2, vend = nfv;
            estart = closestOnly ? nfe - 1 : nfe - 2, eend = nfe;
        }
        for (size_t i = so + 3 * vstart      ; i < so + 3 * vend      ; ++i) visitor(i);
        for (size_t i = so + 3 * nfv + estart; i < so + 3 * nfv + eend; ++i) visitor(i);

        if (restLenVar) {
            assert(closestOnly);
            // Influenced segment rest length
            visitor(um.restLenDofOffsetForSegment(si) + estart);
        }
    };
    if (!restLenVar) {
        if (joint_var_ind < numBaseDoF()) {
            // Joint position, omega, alpha affect all incident segment variables
            for (size_t i = 0; i < num_A_segments + num_B_segments; ++i) visitSegment(m_segments[i], m_isStart[i], false);
        } else { // Edge length A affects the closest two vertices/thetas of rod A only
            visitSegment(getSegmentAt(joint_var_ind - numBaseDoF()), getIsStartAt(joint_var_ind - numBaseDoF()), false);
        }
    }
    else {
        // The joint's rest lengths affect the closest vertex/theta/restkappa/restlen of the corresponding incident segments
        // Since each joint controls the overlapping edge of adjacent segments, we need to check whether the joint is at the start or end of the segment. 
        const size_t lsi = getSegmentIndexForArmAt(joint_var_ind);
        visitSegment(getSegmentAt(lsi), getIsStartAt(lsi), true);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Rod segment operations
////////////////////////////////////////////////////////////////////////////////
// Construct the initial rest points for a rod; note that the endpoints will be
// repositioned if the rod connects at a joint.
template<typename Real_>
std::vector<Pt3_T<Real_>> constructInitialRestPoints(const Pt3_T<Real_> &startPt, const Pt3_T<Real_> &endPt, size_t nsubdiv) {
    if (nsubdiv < 5)
        throw std::runtime_error("Rods in a linkage must have at least 5 edges (to prevent conflicting start/end joint constraints and fully separate joint influences in Hessian)");
    // Half an edge will extend past each endpoint, so only (nsubdiv - 1) edges
    // fit between the endpoints.
    std::vector<Pt3_T<Real_>> rodPts;
    for (size_t i = 0; i <= nsubdiv; ++i) {
        Real_ alpha = (i - 0.5) / (nsubdiv - 1);
        rodPts.push_back((1 - alpha) * startPt + alpha * endPt);
    }
    return rodPts;
}

template<typename Real_>
UmbrellaMesh_T<Real_>::RodSegment::RodSegment(const Pt3 &startPt, const Pt3 &endPt, size_t nsubdiv, SegmentType sType)
    : rod(constructInitialRestPoints(startPt, endPt, nsubdiv)), m_segmentType(sType) { }

template<typename Real_>
template<class Derived>
void UmbrellaMesh_T<Real_>::RodSegment::unpackParameters(const Eigen::DenseBase<Derived> &vars,
                                                       std::vector<Pt3  > &points,
                                                       std::vector<Real_> &thetas) const {
    if (numDoF() > size_t(vars.size())) throw std::runtime_error("DoF indices out of bounds");
    const size_t nv = rod.numVertices(), ne = rod.numEdges();
    points.resize(nv);
    thetas.resize(ne);

    size_t offset = 0;

    // Set the centerline position degrees of freedom
    for (size_t i = 0; i < nv; ++i) {
        // The first/last edge don't contribute degrees of freedom if they're part of a joint.
        if ((i <       2) && (startJoint != NONE)) continue;
        if ((i >= nv - 2) && (endJoint   != NONE)) continue;
        points[i] = vars.template segment<3>(offset);
        offset += 3;
    }

    // Unpack the material axis degrees of freedom
    for (size_t j = 0; j < ne; ++j) {
        if ((j ==      0) && (startJoint != NONE)) continue;
        if ((j == ne - 1) && (endJoint   != NONE)) continue;
        thetas[j] = vars[offset++];
    }
}


template<typename Real_>
template<class Derived>
void UmbrellaMesh_T<Real_>::RodSegment::setParameters(const Eigen::DenseBase<Derived> &vars) {
    auto points = rod.deformedPoints();
    auto thetas = rod.thetas();
    unpackParameters(vars, points, thetas);
    rod.setDeformedConfiguration(points, thetas);
}

template<typename Real_>
template<class Derived>
void UmbrellaMesh_T<Real_>::RodSegment::getParameters(Eigen::DenseBase<Derived> &vars) const {
    if (numDoF() > size_t(vars.size())) throw std::runtime_error("DoF indices out of bounds");
    const auto &pts    = rod.deformedPoints();
    const auto &thetas = rod.thetas();
    const size_t nv = rod.numVertices();
    size_t offset = 0;

    // get the centerline position degrees of freedom
    const size_t first_free_vtx = (startJoint == NONE) ? 0 : 2;
    const size_t  last_free_vtx = (  endJoint == NONE) ? nv - 1 : nv - 3;
    for (size_t i = first_free_vtx; i <= last_free_vtx; ++i) {
        vars.template segment<3>(offset) = pts[i];
        offset += 3;
    }

    // Unpack the material axis degrees of freedom
    const size_t first_free_edge = (startJoint == NONE) ? 0 : 1;
    const size_t nfe = numFreeEdges();
    vars.segment(offset, nfe) = Eigen::Map<const VecX>(&thetas[first_free_edge], nfe);
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::RodSegment::setMinimalTwistThetas(bool verbose) {
    // Minimize twisting energy wrt theta.
    // The twisting energy is quadratic wrt theta, so we simply solve for the
    // step bringing the gradient to zero using the equation:
    //      H dtheta = -g,
    // where g and H are the gradient and Hessian of the twisting energy with
    // respect to material axis angles.
    if ((startJoint == NONE) && (endJoint == NONE))
        throw std::runtime_error("Rod with two free ends--system will be rank deficient");

    const size_t ne = rod.numEdges();

    auto pts = rod.deformedPoints();
    auto ths = rod.thetas();

    // First, remove any unnecessary twist stored in the rod by rotating the second endpoint
    // by an integer multiple of 2PI (leaving d2 unchanged).
    Real_ rodRefTwist = 0;
    const auto &dc = rod.deformedConfiguration();
    for (size_t j = 1; j < ne; ++j)
        rodRefTwist += dc.referenceTwist[j];
    const size_t lastEdge = ne - 1;
    Real_ desiredTheta = ths[0] - rodRefTwist;
    // Probably could be implemented with an fmod...
    while (ths[lastEdge] - desiredTheta >  M_PI) ths[lastEdge] -= 2 * M_PI;
    while (ths[lastEdge] - desiredTheta < -M_PI) ths[lastEdge] += 2 * M_PI;

    if (verbose) {
        std::cout << "rodRefTwist: "         << rodRefTwist        << std::endl;
        std::cout << "desiredTheta: "        << desiredTheta       << std::endl;
        std::cout << "old last edge theta: " << dc.theta(lastEdge) << std::endl;
        std::cout << "new last edge theta: " << ths[lastEdge]      << std::endl;
    }

    rod.setDeformedConfiguration(pts, ths);

    auto H = rod.hessThetaEnergyTwist();
    auto g = rod.gradEnergyTwist();
    std::vector<Real_> rhs(ne);
    for (size_t j = 0; j < ne; ++j)
        rhs[j] = -g.gradTheta(j);

    if (startJoint != NONE) H.fixVariable(       0, 0);
    if (  endJoint != NONE) H.fixVariable(lastEdge, 0);

    auto thetaStep = H.solve(rhs);

    for (size_t j = 0; j < ths.size(); ++j)
        ths[j] += thetaStep[j];

    rod.setDeformedConfiguration(pts, ths);
}

////////////////////////////////////////////////////////////////////////////////
// Hessian matvec implementation
////////////////////////////////////////////////////////////////////////////////
#include "UmbrellaMeshHessVec.inl"


////////////////////////////////////////////////////////////////////////////////
// TerminalEdgeSensitivity 
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
struct UmbrellaMeshTerminalEdgeSensitivity;

template<typename Real_>
const UmbrellaMeshTerminalEdgeSensitivity<Real_> &UmbrellaMesh_T<Real_>::getTerminalEdgeSensitivity(size_t si, TerminalEdge which, bool updatedSource, bool evalHessian) {
    m_sensitivityCache.update(*this, updatedSource, evalHessian);
    assert(si < numSegments());
    assert(m_segments[si].joint(static_cast<int>(which)) != NONE);
    return m_sensitivityCache.lookup(si, which);
}

template<typename Real_>
const UmbrellaMeshTerminalEdgeSensitivity<Real_> &UmbrellaMesh_T<Real_>::getTerminalEdgeSensitivity(size_t si, TerminalEdge which, bool updatedSource, const VecX &delta_params) {
    m_sensitivityCache.update(*this, updatedSource, delta_params);
    assert(si < numSegments());
    assert(m_segments[si].joint(static_cast<int>(which)) != NONE);
    return m_sensitivityCache.lookup(si, which);
}

// Out-of-line constructor and destructor needed because UmbrellaMeshTerminalEdgeSensitivity<Real_> is an incomplete type upon declaration of sensitivityForTerminalEdge.
template<typename Real_> UmbrellaMesh_T<Real_>::SensitivityCache:: SensitivityCache() { }
template<typename Real_> UmbrellaMesh_T<Real_>::SensitivityCache::~SensitivityCache() { }

template<typename Real_>
void UmbrellaMesh_T<Real_>::SensitivityCache::clear() { sensitivityForTerminalEdge.clear(); evaluatedHessian = false; evaluatedWithUpdatedSource = true; }

template<typename Real_>
void UmbrellaMesh_T<Real_>::SensitivityCache::update(const UmbrellaMesh_T &um, bool updatedSource, bool evalHessian) {
    if (evalHessian && !updatedSource) throw std::runtime_error("Hessian formulas only accurate if source frames are updated");
    if (!sensitivityForTerminalEdge.empty() && (evaluatedWithUpdatedSource == updatedSource) && (evaluatedHessian || !evalHessian)) return;
    evaluatedWithUpdatedSource = updatedSource;
    evaluatedHessian = evalHessian;
    const size_t ns = um.numSegments();
    sensitivityForTerminalEdge.resize(2 * ns);
    auto processSegment = [this, evalHessian, updatedSource, &um](size_t si) {
        const auto &s = um.segment(si);
        size_t ji = s.joint(0); if (ji != NONE) sensitivityForTerminalEdge[2 * si + 0].update(um.joint(ji), si, s.rod, updatedSource, evalHessian);
               ji = s.joint(1); if (ji != NONE) sensitivityForTerminalEdge[2 * si + 1].update(um.joint(ji), si, s.rod, updatedSource, evalHessian);
    };
    parallel_for_range(ns, processSegment);
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::SensitivityCache::update(const UmbrellaMesh_T &um, bool updatedSource, const VecX &delta_params) {
    if (!updatedSource) throw std::runtime_error("Hessian formulas only accurate if source frames are updated");
    if (size_t(delta_params.size()) < um.numDoF()) throw std::runtime_error("Input delta params doesn't cover all variables!");
    // If the full joint Hessian is cached and up-to-date, use it to compute the directional derivatives
    const bool validHessianCache = !sensitivityForTerminalEdge.empty() && (evaluatedWithUpdatedSource == updatedSource) && evaluatedHessian;
    evaluatedWithUpdatedSource = updatedSource;
    evaluatedHessian           = validHessianCache; // We only keep the cached Hessian if it is still valid. We do not cache a new one.
    const size_t ns = um.numSegments();
    sensitivityForTerminalEdge.resize(2 * ns);
    auto processSegment = [this, updatedSource, validHessianCache, &um, &delta_params](size_t si) {
        const auto &s = um.segment(si);
        for (size_t lji = 0; lji < 2; ++lji) {
            const size_t ji = s.joint(lji);
            if (ji == NONE) continue;
            const size_t offset = um.dofOffsetForJoint(ji) + 3; // DoF index for first omega variable
            // Warning: tes could be uninitialized at this point!
            auto &tes = sensitivityForTerminalEdge[2 * si + lji];
            // Need to extract the perturbation for the correct length variable from the full perturbation vector.
            Eigen::Matrix<Real_, JointJacobianCols, 1> delta_jparams = delta_params.template segment<JointJacobianCols>(offset);
            size_t localSegmentIndex;
            bool is_A, isStart;
            std::tie(is_A, isStart, localSegmentIndex) = um.joint(ji).terminalEdgeIdentification(si);
            delta_jparams[JointJacobianCols - 1] = delta_params(offset + JointJacobianCols - 1 + localSegmentIndex);
            if (validHessianCache) {
                // tes.update(um.joint(ji), si, s.rod, updatedSource, true);
                for (size_t row_i = 0; row_i < JointJacobianRows;  ++row_i)
                    tes.delta_jacobian.row(row_i) = delta_jparams.transpose() * tes.hessian[row_i];
                // auto delta_jacobian_from_hess = tes.delta_jacobian;
                // tes.update(um.joint(ji), si, s.rod, updatedSource, true, delta_jparams);
                // std::cout << "delta_jacobian mismatch: " << std::endl << tes.delta_jacobian - delta_jacobian_from_hess << std::endl;
                // std::cout << "delta_jacobian: " << std::endl << tes.delta_jacobian << std::endl;
                // std::cout << "delta_from hessian: " << std::endl << delta_jacobian_from_hess << std::endl;
                // std::cout << std::endl << std::endl;
                // if ((tes.delta_jacobian - delta_jacobian_from_hess).norm() / delta_jacobian_from_hess.norm() > 1e-14) {
                //     std::cout << "hessvec error: " << ((tes.delta_jacobian - delta_jacobian_from_hess).norm() / delta_jacobian_from_hess.norm()) << std::endl;
                //     throw std::runtime_error("hessvec error");
                // }
            }
            else {
                tes.update(um.joint(ji), si, s.rod, updatedSource, true, delta_jparams);
            }
        }
    };
    parallel_for_range(ns, processSegment);
}


template<typename Real_>
const UmbrellaMeshTerminalEdgeSensitivity<Real_> & UmbrellaMesh_T<Real_>::SensitivityCache::lookup(size_t si, TerminalEdge which) const {
    return sensitivityForTerminalEdge.at(2 * si + static_cast<int>(which));
}

////////////////////////////////////////////////////////////////////////////////
// Stress analysis
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::gradSurfaceStressLpNormPerEdgeRestLen(CrossSectionStressAnalysis::StressType type, double p, bool updatedSource, bool takeRoot) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".gradSurfaceStressLpNorm");
    VecX_T<Real_> g;
    g.setZero(numExtendedDoF());
    m_assembleSegmentGradient([&](const RodSegment &s) { return s.rod.gradSurfaceStressLpNorm(type, p, updatedSource, /* takeRoot = */ false); },
                              g, updatedSource, /* variableDesignParameters = */ true, /* designParameterOnly = */ false);
    if (takeRoot)
        g *= (1 / p) * pow(surfaceStressLpNorm(type, p, /* takeRoot = */ false), Real_(1 / p) - 1);
    return g;
}


template<typename Real_>
VecX_T<Real_> UmbrellaMesh_T<Real_>::gradSurfaceStressLpNorm(CrossSectionStressAnalysis::StressType type, double p, bool updatedSource, bool takeRoot) const {
    auto gPerEdgeRestLen = gradSurfaceStressLpNormPerEdgeRestLen(type, p, updatedSource, takeRoot);
    VecX result(numExtendedDoFPARL());
    result.head(numDoF()) = gPerEdgeRestLen.head(numDoF());
    if (m_umbrella_dPC.restLen) m_armRestLenToEdgeRestLenMapTranspose.applyRaw(gPerEdgeRestLen.tail(numRestLengths()).data(), result.tail(numArmSegments()).data(), /* no transpose */ false);
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// TargetSurfaceFitter 
////////////////////////////////////////////////////////////////////////////////

struct TargetSurfaceFitter; //Defined in TargetSurfaceFitter.hh

template<typename Real_>
Real_ UmbrellaMesh_T<Real_>::energyAttraction() const {
    // BENCHMARK_SCOPED_TIMER_SECTION timer("energyAttraction");
    if (hasTargetSurface())
        return m_attraction_weight / (m_l0 * m_l0) * m_target_surface_fitter->objective(*this);
    else if (m_attraction_weight != 0) 
        std::cout<<"Attraction Energy Warning: the flag for surface attraction term is active, the attraction weight is non zero, but this Umbrella Mesh has no target surface!"<<std::endl;
    return 0.0;
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::addAttractionGradient(VecX &g) const {
    if (hasTargetSurface()) {
        m_target_surface_fitter->accumulateGradient(*this, g, m_attraction_weight / (m_l0 * m_l0));
    } else if (m_attraction_weight != 0) {
        std::cout<<"Attraction Gradient Warning: the flag for surface attraction term is active, the attraction weight is non zero, but this Umbrella Mesh has no target surface!"<<std::endl;
    }
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::addAttractionHessian(CSCMat &H) const {
    if (hasTargetSurface()) {
        size_t num_projection_pos = m_target_surface_fitter->numQueryPt(*this);
        Real weight = (m_attraction_weight / (m_l0 * m_l0));
        for (size_t qi = 0; qi < num_projection_pos; ++qi) {
            std::tuple<std::vector<size_t>, Real> offset_and_weight = m_target_surface_fitter->getUmbrellaDoFOffsetAndWeightForQueryPt(*this, qi);
            std::vector<size_t> offsets = std::get<0>(offset_and_weight);
            Real query_weight = std::get<1>(offset_and_weight);

            for (size_t ji = 0; ji < offsets.size(); ++ji) {
                H.addDiagBlock(offsets[ji], weight * query_weight * query_weight * m_target_surface_fitter->pt_project_hess(qi));
                if (qi >= numXJoints() && (offsets[ji] < offsets[1 - ji])) {
                    H.addNZBlock(offsets[ji], offsets[1 - ji], weight * query_weight * query_weight * m_target_surface_fitter->pt_project_hess(qi)); 
                }
            }
        }

        for (size_t qi = 0; qi < num_projection_pos; ++qi) {
            std::tuple<std::vector<size_t>, Real> offset_and_weight = m_target_surface_fitter->getUmbrellaDoFOffsetAndWeightForQueryPt(*this, qi);
            std::vector<size_t> offsets = std::get<0>(offset_and_weight);
            Real query_weight = std::get<1>(offset_and_weight);

            for (size_t ji = 0; ji < offsets.size(); ++ji) {
                for (size_t c = 0; c < 3; ++c)
                    H.addDiagEntry(offsets[ji] + c, weight * query_weight * query_weight * m_target_surface_fitter->W_diag_joint_pos[3 * qi + c]);

                if (qi >= numXJoints() && (offsets[ji] < offsets[1 - ji])) {
                    H.addNZBlock(offsets[ji], offsets[1 - ji], weight * query_weight * query_weight * m_target_surface_fitter->pt_tgt_hess(qi));
                }
            }
        }
    } else if (m_attraction_weight != 0) {
        std::cout<<"Attraction Hessian Warning: the flag for surface attraction term is active, the attraction weight is non zero, but this Umbrella Mesh has no target surface!"<<std::endl;
    }
}

// Barier term penalizing violations of the angle penalty
template<typename Real_>
Real_ UmbrellaMesh_T<Real_>::energyAnglePenalty() const {
    if (m_angleBoundEnforcement != AngleBoundEnforcement::Penalty) return 0.0;
    Real_ result = 0.0;
    visitAngleBounds([&](size_t ji, Real_ lower, Real_ upper) {
            result += m_constraintBarrier.eval(joint(ji).alpha(), lower, upper);
        });
    return result;
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::addAnglePenaltyGradient(VecX &g) const {
    if (m_angleBoundEnforcement != AngleBoundEnforcement::Penalty) return;
    visitAngleBounds([&](size_t ji, Real_ lower, Real_ upper) {
            size_t var = m_dofOffsetForJoint[ji] + 6;
            g[var] += m_constraintBarrier.deval(joint(ji).alpha(), lower, upper);
        });
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::addAnglePenaltyHessian(CSCMat &H) const {
    if (m_angleBoundEnforcement != AngleBoundEnforcement::Penalty) return;
    visitAngleBounds([&](size_t ji, Real_ lower, Real_ upper) {
            size_t var = m_dofOffsetForJoint[ji] + 6;
            H.addNZ(var, var, m_constraintBarrier.d2eval(joint(ji).alpha(), lower, upper));
        });
}


template<typename Real_>
bool UmbrellaMesh_T<Real_>::getHoldClosestPointsFixed() const {
    if (hasTargetSurface()) {
        return m_target_surface_fitter->holdClosestPointsFixed;
    } else {
        throw std::runtime_error("This Umbrella Mesh has no target surface! (getHoldClosestPointsFixed)");
    }
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::setHoldClosestPointsFixed(bool holdClosestPointsFixed) {
    if (hasTargetSurface()) {
        m_target_surface_fitter->holdClosestPointsFixed = holdClosestPointsFixed;
    } else {
        std::cout<<"This Umbrella Mesh has no target surface!"<<std::endl;
    }
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::scaleInputPosWeights(Real inputPosWeight, Real bdryMultiplier, Real featureMultiplier, const std::vector<size_t> &additional_feature_pts) {
    if (hasTargetSurface()) {
        m_attraction_input_joint_weight = inputPosWeight;
        m_target_surface_fitter->scaleJointWeights(*this, inputPosWeight, bdryMultiplier, featureMultiplier, additional_feature_pts);
    } else {
        std::cout<<"This Umbrella Mesh has no target surface!"<<std::endl;
    }
    
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::reset_joint_target_with_closest_points() {
    if (hasTargetSurface()) {
        m_target_surface_fitter->reset_joint_target_with_closest_points(*this);
    } else {
        std::cout<<"This Umbrella Mesh has no target surface!"<<std::endl;
    }
}

template<typename Real_>
void UmbrellaMesh_T<Real_>::set_target_surface(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    if (!hasTargetSurface()) m_target_surface_fitter = std::make_shared<TargetSurfaceFitter>();
    // Initialize the target_surface_fitter so we can penalize the umbrella mesh's rigid transformation w.r.t the target surface during compute equilibrium.
    m_target_surface_fitter->scaleJointWeights(*this, m_attraction_input_joint_weight, 1.0, 1.0, std::vector<size_t>(), false);
    m_target_surface_fitter->query_pt_pos_tgt = VecX_T<Real>::Zero(3 * m_target_surface_fitter->numQueryPt(*this));
    m_target_surface_fitter->query_pt_pos_tgt.head(3 * numXJoints()) = stripAutoDiff(XJointTgtPositions());
    m_target_surface_fitter->query_pt_pos_tgt.tail(3 * numUmbrellas()) = stripAutoDiff(UmbrellaMidTgtPositions());

    m_target_surface_fitter->setTargetSurface(*this, V, F);
}
////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation for ordinary double type and autodiff type.
////////////////////////////////////////////////////////////////////////////////
template struct UmbrellaMesh_T<Real>;
template struct UmbrellaMesh_T<ADReal>;

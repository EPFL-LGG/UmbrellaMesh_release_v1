////////////////////////////////////////////////////////////////////////////////
// UmbrellaMesh.hh
////////////////////////////////////////////////////////////////////////////////
#ifndef UmbrellaMesh_HH
#define UmbrellaMesh_HH

#include "ElasticRod.hh"

#include <rotation_optimization.hh>
#include <MeshFEM/Parallelism.hh>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <cmath>
#include "UmbrellaMeshIO.hh"
#include "LinearActuator.hh"
#include "ConstraintBarrier.hh"

struct TargetSurfaceFitter; //Defined in TargetSurfaceFitter.hh

template<typename Real_>
struct UmbrellaMeshTerminalEdgeSensitivity; // Defined in UmbrellaMeshTerminalEdgeSensitivity.hh

// Templated to support automatic differentiation types.
template<typename Real_>
struct UmbrellaMesh_T;

using UmbrellaMesh = UmbrellaMesh_T<Real>;

template<typename Real_>
struct UmbrellaMesh_T {
	using Vec3   = Vec3_T<Real_>;
    using Mat3   = Mat3_T<Real_>;
    using Pt3    =  Pt3_T<Real_>;
    using Vec2   = Vec2_T<Real_>;
    using VecX   = VecX_T<Real_>;
    // This following line should be in MeshFEM Types.hh.
    template<typename Real2_> using Mat3X_T = Eigen::Matrix<Real2_, 3, Eigen::Dynamic>;
    using Mat3X    = Mat3X_T<Real_>;

    using MX3d = Eigen::Matrix<Real_, Eigen::Dynamic, 3>;
    
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;
    using ropt   = rotation_optimization<Real_>;
    using Rod    = ElasticRod_T<Real_>;
    using RealType = Real_;
    using TMatrix = TripletMatrix<Triplet<Real_>>;
    using EnergyType  = typename Rod::EnergyType;
    using umbrella_dPC = DesignParameterConfig;

    enum class UmbrellaEnergyType { Full, Elastic, Deployment, Repulsion, Attraction, AngleBoundPenalty };
    using JointType             = UmbrellaMeshIO::JointType;
    using JointPosType          = UmbrellaMeshIO::JointPosType;
    using SegmentType           = UmbrellaMeshIO::SegmentType;
    using ArmSegmentPosType           = UmbrellaMeshIO::ArmSegmentPosType;
    using DeploymentForceType   = UmbrellaMeshIO::DeploymentForceType;
    using AngleBoundEnforcement = UmbrellaMeshIO::AngleBoundEnforcement;

    static constexpr size_t NONE = std::numeric_limits<size_t>::max();
    static constexpr size_t defaultSubdivision = 10;

    struct Joint;
    struct RodSegment;

    ////////////////////////////////////////////////////////////////////////////
    // Initialization.
    ////////////////////////////////////////////////////////////////////////////
    // Construct empty umbrella mesh, to be initialized later by calling set.
    UmbrellaMesh_T() { }

    // Forward all constructor arguments to set(...)
    template<typename... Args>
    UmbrellaMesh_T(Args&&... args) {
        set(std::forward<Args>(args)...);
    }

    // Main set function: initialize the umbrella mesh from a line graph augmented with joint and segment parameters.
    void set(const UmbrellaMeshIO &io, size_t subdivision = defaultSubdivision);

    // Initialize by copying from another umbrella mesh.
    template<typename Real2_>
    void set(const UmbrellaMesh_T<Real2_> &umbrella_mesh) { setState(umbrella_mesh.serialize()); }

    template<typename DeploymentHeight>
    using SerializedState_T = std::tuple<std::vector<Joint>, std::vector<RodSegment>,
                                       DesignParameterConfig, 
                                       RodMaterial, RodMaterial, Real_,
                                       SuiteSparseMatrix,
                                       VecX, std::vector<size_t>, std::vector<size_t>,
                                       VecX,
                                       size_t, size_t, std::vector<size_t>, size_t,
                                       Real_,
                                       std::shared_ptr<TargetSurfaceFitter>, std::string, Real, Real, Real, 
                                       Real_, DeploymentHeight, std::vector<std::vector<size_t>>,
                                       DeploymentForceType, AngleBoundEnforcement, VecX,
                                       std::vector<std::vector<size_t>>, Real_, 
                                       LinearActuators<UmbrellaMesh_T>>;

    using StateV1 = SerializedState_T<Real_>; // Backward-compatible version for before the per-umbrella targets were introduced.
    using StateV2 = SerializedState_T<VecX>;
    using State   = StateV2;

    // Needed for the pickling to work.
    void set(const StateV1 &state) { setState(state); }
    void set(const State   &state) { setState(state); }
    static State serialize(const UmbrellaMesh_T &um) { return um.serialize(); }

    State serialize() const {
        return std::make_tuple(m_joints, m_segments, 
                               m_umbrella_dPC,
                               m_armMaterial, m_plateMaterial, m_initMinRestLen,
                               m_armRestLenToEdgeRestLenMapTranspose,
                               m_perArmRestLen, m_armIndexForSegment, m_segmentIndexForArm, 
                               m_designParametersPARL, 
                               m_numRigidJoints, m_numArmSegments, m_X_joint_indices, m_numURH,
                               m_E0, 
                               m_target_surface_fitter, std::string()/* m_surface_path */, m_l0, m_attraction_input_joint_weight, m_attraction_weight,
                               m_uniformDeploymentEnergyWeight, m_targetDeploymentHeight, m_umbrella_to_top_bottom_joint_map,
                               m_dftype, m_angleBoundEnforcement, m_deploymentEnergyWeight,
                               m_umbrella_connectivity, m_repulsionEnergyWeight,
                               m_linearActuator);
    }

    template<class SState>
    static std::unique_ptr<UmbrellaMesh_T> deserialize(const SState &state) { return std::make_unique<UmbrellaMesh_T>(state); }

    template<class SState>
    void setState(const SState &state) {
        using Real2_ = std::tuple_element_t<5, SState>; // type of m_initMinRestLen...

        m_armMaterial                                  = std::get< 3>(state);
        m_plateMaterial                                = std::get< 4>(state);
        m_initMinRestLen                               = std::get< 5>(state);
        m_armRestLenToEdgeRestLenMapTranspose          = std::get< 6>(state);
        m_perArmRestLen                                = std::get< 7>(state);
        m_armIndexForSegment                           = std::get< 8>(state);
        m_segmentIndexForArm                           = std::get< 9>(state);
        m_designParametersPARL                         = std::get<10>(state);
        m_numRigidJoints                               = std::get<11>(state);
        m_numArmSegments                               = std::get<12>(state);
        m_X_joint_indices                              = std::get<13>(state);
        m_numURH                                       = std::get<14>(state);
        m_E0                             = stripAutoDiff(std::get<15>(state));
        m_target_surface_fitter                        = std::get<16>(state)->clone(); // Copy! Don't share closest-point state when copying another umbrella mesh.
        // m_surface_path                              = std::get<17>(state);
        m_l0                                           = std::get<18>(state);
        m_attraction_input_joint_weight                = std::get<19>(state);
        m_attraction_weight                            = std::get<20>(state);
        m_uniformDeploymentEnergyWeight                = std::get<21>(state);
        auto deploymentHeights                         = std::get<22>(state);
        m_umbrella_to_top_bottom_joint_map             = std::get<23>(state);
        m_dftype                                       = std::get<24>(state);
        m_angleBoundEnforcement                        = std::get<25>(state);
        m_deploymentEnergyWeight                       = std::get<26>(state);
        m_umbrella_connectivity                        = std::get<27>(state);
        m_repulsionEnergyWeight                        = std::get<28>(state);
        m_linearActuator                               = std::get<29>(state);

        // Set per-umbrella heights either to the passed vector (V2) or set all deployment heights to a single scalar (V1)
        const size_t nu = m_umbrella_to_top_bottom_joint_map.size();
        m_targetDeploymentHeight.resize(nu);
        m_targetDeploymentHeight.array() = deploymentHeights; 

        set<Real2_>(std::get< 0>(state), std::get< 1>(state), std::get< 2>(state));
    }

    // Set umbrella mesh by copying the passed joints, segments and homogeneous material.
    // Note: "down-casting from autodiff to non-autodiff number type is unsupported!
    template<typename Real2_>
    void set(const std::vector<typename UmbrellaMesh_T<Real2_>::Joint> &joints, const std::vector<typename UmbrellaMesh_T<Real2_>::RodSegment> &segments, const umbrella_dPC &designParameter_config) {
        m_joints.clear();
        m_joints.reserve(joints.size());
        for (const auto &j : joints) m_joints.emplace_back(j);

        m_segments.clear();
        m_segments.reserve(segments.size());
        for (const auto &s : segments) m_segments.emplace_back(s);

        // Update the degree of freedom info/redirect the joints to point at this linkage
        m_buildDoFOffsets();
        for (auto &j : m_joints) { j.updateUmbrellaMeshPointer(this); }

        setDesignParameterConfig(designParameter_config.restLen, designParameter_config.restKappa, true);
    }

    void set_target_surface(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

    // Avoid accidentally copying umbrella meshes around for performance reasons;
    // explicitly use UmbrellaMesh_T::set instead.
    // If we choose to offer this operator in the future, it should be
    // implemented as a call to set (the joint umbrella mesh pointers must be updated)
    UmbrellaMesh_T &operator=(const UmbrellaMesh_T &b) = delete;

    const std::vector<RodSegment> &segments() const { return m_segments; }
          std::vector<RodSegment> &segments()       { return m_segments; }
    const std::vector<Joint>      &joints()   const { return m_joints; }
          std::vector<Joint>      &joints()         { return m_joints; }

    const Joint &joint(size_t i) const { return m_joints.at(i); }
          Joint &joint(size_t i)       { return m_joints.at(i); }

    const RodSegment &segment(size_t i) const { return m_segments.at(i); }
          RodSegment &segment(size_t i)       { return m_segments.at(i); }

    void set_segment(RodSegment new_seg, size_t i) { m_segments.at(i) = new_seg; }

	// Set the same material for every rod in the umbrella mesh.
    void setMaterial(const RodMaterial &material) { setMaterial(material, material); }
    // Use different materials for the "arm" segments from the "plate" segments.
    void setMaterial(const RodMaterial &armMaterial, const RodMaterial &plateMaterial);

    const RodMaterial &  armMaterial() const { return   m_armMaterial; }
    const RodMaterial &plateMaterial() const { return m_plateMaterial; }

    // Set the current adapted curve frame as the source for parallel transport.
    // See ElasticRod::updateSourceFrame for more discussion.
    // Also set each joint's source normal used to encourage temporal coherence
    // of normals as the linkage's opening angle reverses sign.
    void updateSourceFrame() {
        parallel_for_range(numSegments(), [this](size_t si) { segment(si).rod.updateSourceFrame(); });
        m_sensitivityCache.clear();
    }

    // Apply each joint's current rotation to its source frame, resetting the
    // joint rotation variables to the identity.
    // This could be done at every iteration of Newton's method to
    // speed up rotation gradient/Hessian calculation, or only when needed
    // as the rotation magnitude becomes too great (our rotation parametrization
    // has a singularity when the rotation angle hits pi).
    void updateRotationParametrizations() {
        if (disableRotationParametrizationUpdates) return;
        parallel_for_range(numJoints(), [this](size_t ji) { joint(ji).updateParametrization(); });
        m_sensitivityCache.clear();
    }

    // For debugging gradients/Hessians with finite differences, the rotation
    // parametrization update can be confusing since then the rotation variables
    // at equilibrium are always zero. We allow the user to disable these updates
    // for debugging purposes (at the cost of increased computation for the rotation
    // derivatives).
    bool disableRotationParametrizationUpdates = false;

    ////////////////////////////////////////////////////////////////////////////
    // DoFs and Energy Values.
    ////////////////////////////////////////////////////////////////////////////

    size_t numDoF() const;
    size_t numSegments() const { return segments().size(); }
    size_t numJoints()   const { return m_joints.size(); }
    size_t numRigidJoints()   const { return m_numRigidJoints; }
    size_t numCenterlinePos() const { return m_dofOffsetForCenterlinePos.size(); }
    size_t dofOffsetForJoint  (size_t ji) const { return m_dofOffsetForJoint  .at(ji); }
    size_t dofOffsetForSegment(size_t si) const { return m_dofOffsetForSegment.at(si); }
    size_t dofOffsetForCenterlinePos(size_t cli) const {return m_dofOffsetForCenterlinePos.at(cli); }
    size_t restLenDofOffsetForSegment(size_t si) const {return m_restLenDofOffsetForSegment.at(si); }
    size_t designParameterDoFOffsetForJoint(size_t ji) const {return m_designParameterDoFOffsetForJoint.at(ji); }

    Real_ initialMinRestLength() const { return m_initMinRestLen; }

    // Parameter order: all segment parameters, followed by all joint parameters
    VecX getDoFs() const;
    void setDoFs(const Eigen::Ref<const VecX> &dofs, bool spatialCoherence = false);
    void setDoFs(const std::vector<Real_> &dofs) { setDoFs(Eigen::Map<const VecX>(dofs.data(), dofs.size())); }

    ////////////////////////////////////////////////////////////////////////////
    // Extended DoFs for design optimization.
    ////////////////////////////////////////////////////////////////////////////

    size_t numFreeRestLengths() const;
    size_t numJointRestLengths() const;
    VecX getFreeRestLengths() const;
    VecX getJointRestLengths() const;
    size_t numRestLengths() const;
    VecX getRestLengths() const;
    Real_ minRestLength() const { return getRestLengths().minCoeff(); }

    size_t numArmSegments() const { return m_numArmSegments; }
    size_t numURH() const { return m_numURH; }
    size_t numUmbrellas() const { return m_umbrella_to_top_bottom_joint_map.size(); }

    size_t numXJoints() const { return m_X_joint_indices.size(); }

    size_t freeRestLenOffset() const { return numDoF(); }
    size_t designParameterOffset() const { return numDoF(); }
    size_t jointRestLenOffset() const { return numDoF() + numFreeRestLengths() * m_umbrella_dPC.restLen; }


    size_t numExtendedDoF() const { return numDoF() + numRestLengths() * m_umbrella_dPC.restLen; }
    size_t numExtendedDoFPARL() const { return numDoF() + numArmSegments() * m_umbrella_dPC.restLen; }

    VecX getExtendedDoFs() const;
    void setExtendedDoFs(const VecX &params, bool spatialCoherence = false);

    VecX getExtendedDoFsPARL() const;
    void setExtendedDoFsPARL(const VecX &params, bool spatialCoherence = false);

    void setPerArmRestLength(const VecX &parl) { m_perArmRestLen = parl; m_setRestLengthsFromPARL(); m_designParametersPARL.tail(numArmSegments()) = m_perArmRestLen;}
    VecX getPerArmRestLength() const { return m_perArmRestLen; }

    SuiteSparseMatrix armRestLenToEdgeRestLenMapTranspose() const { return m_armRestLenToEdgeRestLenMapTranspose; }

    const DesignParameterConfig &getDesignParameterConfig() const {
        return m_umbrella_dPC;
    }

    void setDesignParameterConfig(bool use_restLen, bool use_restKappa, bool update_designParams_cache = true) {
        m_umbrella_dPC.restLen = use_restLen;
        m_umbrella_dPC.restKappa = use_restKappa;
        for (auto &s : m_segments) s.rod.setDesignParameterConfig(use_restLen, use_restKappa);

        if (update_designParams_cache) {
            VecX designParams(numDesignParams());
            if (m_umbrella_dPC.restKappa) {
                throw std::runtime_error("The umbrella mesh currently has not rest curvature variables!");
            }
            if (m_umbrella_dPC.restLen) designParams.tail(numArmSegments()) = m_perArmRestLen;
            m_designParametersPARL = designParams;
        }

        m_buildDoFOffsets();
        m_clearCache();
        m_sensitivityCache.clear();
    }

    // Need to be updated to use the umbrella heights.
    // Design optimization: currently we optimize for the rest curvature (kappa) and the rest lengths
    const VecX &getDesignParameters() const { 
        return m_designParametersPARL; 
    }

    void setDesignParameters(Eigen::Ref<const VecX> p) {
        m_designParametersPARL = p;
        if (m_umbrella_dPC.restLen) {
            m_perArmRestLen = p.tail(numArmSegments());
            m_setRestLengthsFromPARL();
        }
    }

    // Gradient of elastic energy with respect to the design parameters
    VecX grad_design_parameters(bool updatedSource = false) const {
        auto gPerEdgeRestLen = gradient(updatedSource, UmbrellaEnergyType::Full, EnergyType::Full, true, /* only compute design parameter components (but the vector is still full length, the DoF part is just zeros) */ true);
        VecX result(numDesignParams());
        result.setZero();
        if (m_umbrella_dPC.restLen) m_armRestLenToEdgeRestLenMapTranspose.applyRaw(gPerEdgeRestLen.tail(numRestLengths()).data(), result.tail(numArmSegments()).data(), /* no transpose */ false);
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // DesignOptimization Support
    ////////////////////////////////////////////////////////////////////////////
    size_t numDesignParams() const { return numArmSegments() * m_umbrella_dPC.restLen; }
    size_t numSimVars()      const { return numDoF(); }

    ////////////////////////////////////////////////////////////////////////////
    // Design parameter solve support.
    ////////////////////////////////////////////////////////////////////////////
    size_t designParameterSolve_numDoF()                                                                const { return numExtendedDoFPARL(); }
    size_t designParameterSolve_numDesignParameters()                                                   const { return numExtendedDoFPARL() - numDoF(); }
    VecX designParameterSolve_getDoF()                                                                  const { return getExtendedDoFsPARL(); }
    void designParameterSolve_setDoF(const VecX &params)                                                      { return setExtendedDoFsPARL(params); }

    Real_ designParameterSolve_energy()                                                                 const { return energy(); }
    VecX designParameterSolve_gradient(bool updatedSource = false, UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType eType = EnergyType::Full) const { 
        return gradientPerArmRestlen(updatedSource, type, eType);
    }
    CSCMat designParameterSolve_hessianSparsityPattern()                                                const { return hessianPerArmRestlenSparsityPattern(); }
    void designParameterSolve_hessian(CSCMat &H, UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType etype = EnergyType::Full)                   const { hessianPerArmRestlen(H, type, etype); }
    std::vector<size_t> designParameterSolve_lengthVars() const {
        // first take the joint length variables (Not rest length! Omit the per-edge rest length vars!)
        auto result = lengthVars(false);
        const size_t rlo = freeRestLenOffset();
        for (size_t si = 0; si < numArmSegments(); ++si) result.push_back(rlo + si);
        return result;
    }

    std::vector<size_t> designParameterSolve_restLengthVars() const {
        std::vector<size_t> result;
        const size_t rlo = freeRestLenOffset();
        for (size_t si = 0; si < numArmSegments(); ++si) result.push_back(rlo + si);
        return result;
    }

    // Indices of all variables that must be fixed during the rest length solve
    // This consists of all X joint positions, plus the joint with valence 1.
    std::vector<size_t> designParameterSolveFixedVars() const {
        std::vector<size_t> result;
        const size_t nj = numJoints();
        result.reserve(3 * nj);
        for (size_t i = 0; i < nj; ++i) {
            if (m_joints[i].jointType() == JointType::X || m_joints[i].valence() <= 1) {
                size_t o = m_dofOffsetForJoint[i];
                result.push_back(o + 0);
                result.push_back(o + 1);
                result.push_back(o + 2);
            }
        }
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Energy
    ////////////////////////////////////////////////////////////////////////////

    // Elastic energy stored in the umbrella mesh.
    Real_ energyElastic() const;
    Real_ energyStretch() const;
    Real_ energyBend() const;
    Real_ energyTwist() const;


    Real_ energyElastic(EnergyType type) const {
        switch (type) {
            case EnergyType::   Full: return energyElastic();
            case EnergyType::   Bend: return energyBend();
            case EnergyType::  Twist: return energyTwist();
            case EnergyType::Stretch: return energyStretch();
            default: throw std::runtime_error("Unknown energy type");
        }
    }

    // All energies *but* the elastic energy.
    Real_ energyAuxiliary() const { return energyDeployment() + energyRepulsion() + energyAttraction() + energyAnglePenalty(); }

    Real_ energy(UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType eType = EnergyType::Full) const {
        switch (type) {
            case UmbrellaEnergyType::Full:              return energyElastic(eType)  + energyAuxiliary();
            case UmbrellaEnergyType::Elastic:           return energyElastic(eType);
            case UmbrellaEnergyType::Deployment:        return energyDeployment();
            case UmbrellaEnergyType::Repulsion:         return energyRepulsion();
            case UmbrellaEnergyType::Attraction:        return energyAttraction();
            case UmbrellaEnergyType::AngleBoundPenalty: return energyAnglePenalty();
            default: throw std::runtime_error("Unknown umbrella energy type");
        }
    }
    ////////////////////////////////////////////////////////////////////////////
    // Derivatives.
    ////////////////////////////////////////////////////////////////////////////

    // Gradient of the umbrella mesh's elastic energy with respect to all degrees of freedom.
    // If "updatedSource" is true, we use the more efficient gradient formulas
    // that are only accurate after a call to updateSourceFrame().
    VecX gradientPerArmRestlen(bool updatedSource = false, UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType eType = EnergyType::Full) const;

    VecX gradient(bool updatedSource = false, UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType eType = EnergyType::Full, bool variableDesignParameters = false, bool designParameterOnly = false) const {
        using UET = UmbrellaEnergyType;
        BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".gradient");
        VecX g(variableDesignParameters ? numExtendedDoF() : numDoF());
        g.setZero();

        if (type == UET::Full || type == UET::Elastic) {
            const auto &gradGetter = [&](const RodSegment &s) -> typename ElasticRod_T<Real_>::Gradient {
                    return s.rod.gradient(updatedSource, eType, variableDesignParameters, designParameterOnly);
                };
            m_assembleSegmentGradient(gradGetter, g, updatedSource, variableDesignParameters, designParameterOnly);
        }

        if (type == UET::Full || type == UET::       Deployment) {   addDeploymentGradient(g); }
        if (type == UET::Full || type == UET::        Repulsion) {    addRepulsionGradient(g); }
        if (type == UET::Full || type == UET::       Attraction) {   addAttractionGradient(g); }
        if (type == UET::Full || type == UET::AngleBoundPenalty) { addAnglePenaltyGradient(g); }
        return g;
    }

    // The number of non-zeros in the Hessian's sparsity pattern (a tight
    // upper bound for the number of non-zeros for any configuration).
    size_t hessianNNZ(bool variableDesignParameters = false) const;

    // Optimizers like Knitro and Ipopt need to know all Hessian entries that
    // could ever possibly be nonzero throughout the course of optimization.
    // The current Hessian may be missing some of these entries.
    // Knowing the fixed sparsity pattern also allows us to more efficiently construct the Hessian.
    CSCMat hessianSparsityPattern(bool variableDesignParameters = false, Real_ val = 0.0) const;

    // Accumulate the Hessian into the sparse matrix "H," which must already be initialized
    // with the sparsity pattern.
    void hessian(CSCMat &H, UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType eType = EnergyType::Full, const bool variableDesignParameters = false) const;

    // Hessian of the umbrella mesh's elastic energy with respect to all degrees of freedom.
    TMatrix hessian(UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType eType = EnergyType::Full, const bool variableDesignParameters = false) const;

    CSCMat hessianPerArmRestlenSparsityPattern(Real_ val = 0.0) const;
    void hessianPerArmRestlen(CSCMat &H, UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType etype = EnergyType::Full) const;
    TMatrix hessianPerArmRestlen(UmbrellaEnergyType type = UmbrellaEnergyType::Full, EnergyType eType = EnergyType::Full) const;

    VecX applyHessianElastic(const VecX &v, bool variableDesignParameters, const HessianComputationMask &mask) const;
    VecX applyHessianDeployment(const VecX &v, const HessianComputationMask &mask) const;
    VecX applyHessianAttraction(const VecX &v, const HessianComputationMask &mask) const;
    VecX applyHessian(const VecX &v, bool variableDesignParameters = false, const HessianComputationMask &mask = HessianComputationMask(), UmbrellaEnergyType type = UmbrellaEnergyType::Full) const;

    VecX applyHessianPerArmRestlen(const VecX &v, const HessianComputationMask &mask = HessianComputationMask(), UmbrellaEnergyType type = UmbrellaEnergyType::Full) const;
    
    // useLumped: whether to use the rods' diagonal lumped mass matrices.
    // The assembly's mass matrix will be non-diagonal in either case because the joint
    // parameters control multiple rod centerline point/theta variables.
    void massMatrix(CSCMat &M, bool updatedSource = false, bool useLumped = false) const;
    TMatrix massMatrix(bool updatedSource = false, bool useLumped = false) const {
        auto M = hessianSparsityPattern();
        massMatrix(M, updatedSource, useLumped);
        return M.getTripletMatrix();
    }
    // Probably not useful: this matrix is usually not positive definite
    VecX lumpedMassMatrix(bool updatedSource = false) const;

    // Approximate the greatest velocity of any point in the rod induced by
    // changing the parameters at rate paramVelocity.
    // ***Assumes that the source frame has been updated***.
    Real_ approxLinfVelocity(const VecX &paramVelocity) const;


    /////////////
    // Deployment
    /////////////
    VecX plateHeights() const { return m_linearActuator.plateHeights(*this); }
    Real_ linearActuatorEnergy() const {
        return m_linearActuator.energy(*this, m_deploymentEnergyWeight);
    }

    VecX linearActuatorGradient() const {
        VecX g;
        g.setZero(numDoF());
        m_linearActuator.addGradient(*this, m_deploymentEnergyWeight, g);
        return g;
    }

    CSCMat linearActuatorHessian() const {
        CSCMat H(hessianSparsityPattern(false));
        m_linearActuator.addHessian(*this, m_deploymentEnergyWeight, H);
        return H;
    }

    VecX linearActuatorHessVec(const VecX &v) const {
        VecX delta_g;
        delta_g.setZero(numDoF());
        m_linearActuator.addHessVec(*this, m_deploymentEnergyWeight, v, delta_g);
        return delta_g;
    }

    Real_ energyDeployment() const {
        if (m_dftype == DeploymentForceType::LinearActuator) return linearActuatorEnergy();
        // Debug
        if (checkUmbrellaFlipped()) return safe_numeric_limits<Real_>::max();
        // BENCHMARK_SCOPED_TIMER_SECTION timer("energyDeployment");
        VecX umbrellaStrain = getUmbrellaHeights().array() - m_targetDeploymentHeight.array();
        if (m_dftype == DeploymentForceType::Constant) return umbrellaStrain.cwiseAbs().dot(m_deploymentEnergyWeight);
        if (m_dftype == DeploymentForceType::  Spring) return 0.5 * umbrellaStrain.dot(m_deploymentEnergyWeight.cwiseProduct(umbrellaStrain));
		throw std::runtime_error("Unknown DeploymentForceType");
    }

    void addDeploymentGradient(VecX &g) const {
        if (m_dftype == DeploymentForceType::LinearActuator) {
            m_linearActuator.addGradient(*this, m_deploymentEnergyWeight, g);
            return;
        }
        VecX umbrellaHeights = getUmbrellaHeights();
        for (size_t ui = 0; ui < numUmbrellas(); ++ui) {
            size_t top_ji = getUmbrellaCenterJi(ui, 0);
            size_t bot_ji = getUmbrellaCenterJi(ui, 1);
            Real_ coeff = (umbrellaHeights[ui] - m_targetDeploymentHeight[ui]);
            if (m_dftype == DeploymentForceType::Constant) {
                if(coeff != 0) coeff /= abs(coeff);
            }
            coeff *= m_deploymentEnergyWeight[ui];
            Vec3 tangent = (joint(top_ji).pos() - joint(bot_ji).pos()).normalized();
            g.template segment<3>(dofOffsetForJoint(top_ji)) +=   coeff * tangent;
            g.template segment<3>(dofOffsetForJoint(bot_ji)) += - coeff * tangent;
        }
    }

    void addDeploymentHessian(CSCMat &H) const {
        if (m_dftype == DeploymentForceType::LinearActuator) {
            m_linearActuator.addHessian(*this, m_deploymentEnergyWeight, H);
            return;
        }
        using M3d = Mat3_T<Real_>;
        VecX umbrellaHeights = getUmbrellaHeights();
        for (size_t ui = 0; ui < numUmbrellas(); ++ ui) {
            size_t top_ji = getUmbrellaCenterJi(ui, 0);
            size_t bot_ji = getUmbrellaCenterJi(ui, 1);
            size_t top_ji_dof = dofOffsetForJoint(top_ji);
            size_t bot_ji_dof = dofOffsetForJoint(bot_ji);
            Real_ height = umbrellaHeights[ui];
            if (height == 0) std::cout<<"Distance Deployment Hessian term encountered zero height umbrella at joint "<<ui<<"!"<<std::endl;
            
            Real_ coeff_one = m_deploymentEnergyWeight[ui] - m_deploymentEnergyWeight[ui] * m_targetDeploymentHeight[ui] / height;
            Real_ coeff_two = m_deploymentEnergyWeight[ui] - coeff_one;
            Vec3 tangent = (joint(top_ji).pos() - joint(bot_ji).pos()) / height;
            M3d hessianBlock;
            if (m_dftype == DeploymentForceType::Constant) {
                hessianBlock = coeff_one * M3d::Identity() - (coeff_one * tangent) * tangent.transpose();
                if (height - m_targetDeploymentHeight[ui] != 0) hessianBlock /= abs(height - m_targetDeploymentHeight[ui]);
            }
            else hessianBlock = coeff_one * M3d::Identity() + (coeff_two * tangent) * tangent.transpose();
            
            for (size_t i = 0; i < 3; ++i) {
                H.addNZStrip(top_ji_dof, top_ji_dof + i, hessianBlock.col(i).head(i+1));
            }
            for (size_t i = 0; i < 3; ++i) {
                H.addNZStrip(bot_ji_dof, bot_ji_dof + i, hessianBlock.col(i).head(i+1));
            }
            if (top_ji_dof <= bot_ji_dof) {
                for (size_t i = 0; i < 3; ++i) {
                    H.addNZStrip(top_ji_dof, bot_ji_dof + i, - hessianBlock.col(i));
                }
            } 
            else {
                for (size_t i = 0; i < 3; ++i) {
                    H.addNZStrip(bot_ji_dof, top_ji_dof + i,    - hessianBlock.col(i));
                }
            }
        }
    }

    bool checkUmbrellaFlipped() const {
        const size_t nu = numUmbrellas();
        for (size_t ui = 0; ui < nu; ++ui) {
            Vec3 umbrellaVec = (m_joints[getUmbrellaCenterJi(ui, 0)].pos() -
                          m_joints[getUmbrellaCenterJi(ui, 1)].pos());
            if ((m_joints[getUmbrellaCenterJi(ui, 0)].ghost_normal().dot(umbrellaVec) < 0) ||
                (m_joints[getUmbrellaCenterJi(ui, 1)].ghost_normal().dot(umbrellaVec) > 0)) {
                std::cout<<"Warning: Umbrella "<<ui<<" is flipped!"<<std::endl;
                return true;
            }
        }
        return false;
    }

    /////////////
    // Repulsion
    /////////////

    const std::vector<size_t> top_neighbor_ji(size_t ji) const {
        std::vector<size_t> result;
        for (size_t eid = 0; eid < m_umbrella_connectivity.size(); ++eid) {
            size_t i = m_umbrella_connectivity[eid][0], j = m_umbrella_connectivity[eid][1];
            size_t jid_i = m_umbrella_to_top_bottom_joint_map[i][0], jid_j = m_umbrella_to_top_bottom_joint_map[j][0];
            if(jid_i == ji) result.push_back(jid_j);
            if(jid_j == ji) result.push_back(jid_i);
        }
        return result;
    }
    const std::vector<size_t> bot_neighbor_ji(size_t ji) const {
        std::vector<size_t> result;
        for (size_t eid = 0; eid < m_umbrella_connectivity.size(); ++eid) {
            size_t i = m_umbrella_connectivity[eid][0], j = m_umbrella_connectivity[eid][1];
            size_t jid_i = m_umbrella_to_top_bottom_joint_map[i][1], jid_j = m_umbrella_to_top_bottom_joint_map[j][1];
            if(jid_i == ji) result.push_back(jid_j);
            if(jid_j == ji) result.push_back(jid_i);
        }
        return result;
    }

    Real_ energyRepulsion() const {
        // BENCHMARK_SCOPED_TIMER_SECTION timer("energyRepulsion");
        VecX topPos = topJointPositions();
        VecX botPos = bottomJointPositions();

        Real_ energy = 0;
        for (size_t eid = 0; eid < m_umbrella_connectivity.size(); ++eid) {
            size_t i = m_umbrella_connectivity[eid][0], j = m_umbrella_connectivity[eid][1];
            // i <-- j
            Vec3 top_tangent = (topPos.template segment<3>(3 * i)) - (topPos.template segment<3>(3 * j));
            Vec3 bot_tangent = (botPos.template segment<3>(3 * i)) - (botPos.template segment<3>(3 * j));
            energy += -m_repulsionEnergyWeight * (top_tangent.norm() + bot_tangent.norm());
        }
        return energy;
    }

    void addRepulsionGradient(VecX &g) const {
        VecX topPos = topJointPositions();
        VecX botPos = bottomJointPositions();

        for (size_t eid = 0; eid < m_umbrella_connectivity.size(); ++eid) {
            size_t i = m_umbrella_connectivity[eid][0], j = m_umbrella_connectivity[eid][1];
            size_t jid_i_t = m_umbrella_to_top_bottom_joint_map[i][0], jid_j_t = m_umbrella_to_top_bottom_joint_map[j][0];
            size_t jid_i_b = m_umbrella_to_top_bottom_joint_map[i][1], jid_j_b = m_umbrella_to_top_bottom_joint_map[j][1];
            // i <-- j
            Vec3 top_tangent = (topPos.template segment<3>(3 * i)) - (topPos.template segment<3>(3 * j));
            Vec3 bot_tangent = (botPos.template segment<3>(3 * i)) - (botPos.template segment<3>(3 * j));
            top_tangent /= top_tangent.norm();
            bot_tangent /= bot_tangent.norm();

            g.template segment<3>(dofOffsetForJoint(jid_j_t)) -= -m_repulsionEnergyWeight * top_tangent;
            g.template segment<3>(dofOffsetForJoint(jid_i_t)) += -m_repulsionEnergyWeight * top_tangent;
            g.template segment<3>(dofOffsetForJoint(jid_j_b)) -= -m_repulsionEnergyWeight * bot_tangent;
            g.template segment<3>(dofOffsetForJoint(jid_i_b)) += -m_repulsionEnergyWeight * bot_tangent;
        }
    }

    
    // Hessian Helper
    size_t numRepulsionNNZ() const {
        // Add the interaction between the each repulsion connection
        return m_umbrella_connectivity.size() * 9 * 2;
    }
    void addRepulsionHessian(CSCMat &H) const {
        VecX topPos = topJointPositions();      
        VecX botPos = bottomJointPositions();
        using M3d = Mat3_T<Real_>;
        for (size_t eid = 0; eid < m_umbrella_connectivity.size(); ++eid) {
            size_t i = m_umbrella_connectivity[eid][0], j = m_umbrella_connectivity[eid][1];
            size_t jid_i_t = m_umbrella_to_top_bottom_joint_map[i][0], jid_j_t = m_umbrella_to_top_bottom_joint_map[j][0];
            size_t jid_i_b = m_umbrella_to_top_bottom_joint_map[i][1], jid_j_b = m_umbrella_to_top_bottom_joint_map[j][1];
            // i <-- j
            Vec3 top_tangent = (topPos.template segment<3>(3 * i)) - (topPos.template segment<3>(3 * j));
            Vec3 bot_tangent = (botPos.template segment<3>(3 * i)) - (botPos.template segment<3>(3 * j));
            Real_ top_distance = top_tangent.norm(), bot_distance = bot_tangent.norm();
            top_tangent /= top_tangent.norm();
            bot_tangent /= bot_tangent.norm();
            
            size_t i_t_dof = dofOffsetForJoint(jid_i_t), j_t_dof = dofOffsetForJoint(jid_j_t);
            size_t i_b_dof = dofOffsetForJoint(jid_i_b), j_b_dof = dofOffsetForJoint(jid_j_b);
            
            M3d hessianBlockTop = (M3d::Identity() - top_tangent * top_tangent.transpose())/top_distance;
            M3d hessianBlockBot = (M3d::Identity() - bot_tangent * bot_tangent.transpose())/bot_distance;
            hessianBlockTop *= -m_repulsionEnergyWeight;
            hessianBlockBot *= -m_repulsionEnergyWeight;

            for (size_t i = 0; i < 3; ++i) {
                H.addNZStrip(i_t_dof,    i_t_dof + i,      hessianBlockTop.col(i).head(i+1));
                H.addNZStrip(i_b_dof,    i_b_dof + i,      hessianBlockBot.col(i).head(i+1));
            }
            for (size_t i = 0; i < 3; ++i) {
                H.addNZStrip(j_t_dof, j_t_dof + i,   hessianBlockTop.col(i).head(i+1));
                H.addNZStrip(j_b_dof, j_b_dof + i,   hessianBlockBot.col(i).head(i+1));
            }
            if (i_t_dof <= j_t_dof) {
                for (size_t i = 0; i < 3; ++i) {
                    H.addNZStrip(i_t_dof,    j_t_dof + i, - hessianBlockTop.col(i));
                }
            } 
            else {
                for (size_t i = 0; i < 3; ++i) {
                    H.addNZStrip(j_t_dof, i_t_dof + i,    - hessianBlockTop.col(i));
                }
            }
            if (i_b_dof <= j_b_dof) {
                for (size_t i = 0; i < 3; ++i) {
                    H.addNZStrip(i_b_dof,    j_b_dof + i, - hessianBlockBot.col(i));
                }
            } 
            else {
                for (size_t i = 0; i < 3; ++i) {
                    H.addNZStrip(j_b_dof, i_b_dof + i,    - hessianBlockBot.col(i));
                }
            }
        }
    }

    /////////////
    // Attraction
    /////////////

    Real_ energyAttraction() const;
    void addAttractionGradient(VecX &g) const;
    void addAttractionHessian(CSCMat &H) const;

    Real_ energyAnglePenalty() const;
    void addAnglePenaltyGradient(VecX &g) const;
    void addAnglePenaltyHessian(CSCMat &H) const;

    bool getHoldClosestPointsFixed() const;
    void setHoldClosestPointsFixed(bool holdClosestPointsFixed);

    void scaleInputPosWeights(Real inputPosWeight, Real bdryMultiplier = 1.0, Real featureMultiplier = 1.0, const std::vector<size_t> &additional_feature_pts = std::vector<size_t>());

    void reset_joint_target_with_closest_points();
    ////////////////////////////////////////////////////////////////////////////
    // Equilibrium Problem Helper Functions.
    ////////////////////////////////////////////////////////////////////////////    
    using BC = NewtonProblem::BoundConstraint;
    std::vector<BC> equilibriumProblemBoundConstraints() const {
        // Make sure joint edge lengths aren't shrunk down to zero/inverted
        const Real lengthVal = stripAutoDiff(0.01 * initialMinRestLength());
        auto lv = lengthVars(false /* rest lengths are fixed */);

        std::vector<BC> result;
        for (size_t var : lv) result.emplace_back(var, lengthVal, BC::Type::LOWER);
        if (m_angleBoundEnforcement == AngleBoundEnforcement::Hard) {
            visitAngleBounds([&](size_t ji, Real lower, Real upper) {
                    size_t var = m_dofOffsetForJoint[ji] + 6;
                    result.emplace_back(var, lower, BC::Type::LOWER);
                    result.emplace_back(var, upper, BC::Type::UPPER);
                });
        }
        return result;
    }

    // Indices of degrees of freedom controlling joint openings.
    std::vector<size_t> jointAngleDoFIndices() const {
        std::vector<size_t> result;
        result.reserve(numJoints() - numRigidJoints());
        for (size_t j = 0; j < numJoints(); ++j) {
            if (m_joints[j].jointType() != JointType::Rigid) result.push_back(m_dofOffsetForJoint[j] + 6); // pos, omega, alpha <--
        }
        return result;
    }

    std::vector<size_t> rigidJointAngleDoFIndices() const {
        std::vector<size_t> result;
        result.reserve(numRigidJoints());
        for (size_t j = 0; j < numJoints(); ++j) {
            if (m_joints[j].jointType() == JointType::Rigid) result.push_back(m_dofOffsetForJoint[j] + 6); // pos, omega, alpha <--
        }
        return result;
    }

    // Compute the average over all joints of the joint opening angle.
    Real_ getAverageJointAngle() const {
        Real_ result = 0;
        for (const auto &j : m_joints) {
            switch (j.jointType()) {
                case JointType::Rigid: break;
                case JointType::X: result += j.alpha(); break;
                case JointType::T: result += 2.0 * j.alpha(); break;
                default: throw std::runtime_error("Unknown joint type");
            }
        }
        return result / (numJoints() - numRigidJoints());
    }

    // Change the average joint opening angle by uniformly scaling all joint openings.
    // (This only changes the angles/incident segment edges. No equilibrium solve is run.)
    void setAverageJointAngle(const Real_ alpha) {
        const Real_ curr = getAverageJointAngle();
        if(curr != 0) {
            const Real_ scale = alpha / curr;
            for (auto &j : m_joints) {
                if (j.jointType() != JointType::Rigid) j.set_alpha(j.alpha() * scale);
            }
        }
        else {
            for (auto &j : m_joints) {
                switch (j.jointType()) {
                    case JointType::Rigid: break;
                    case JointType::X: j.set_alpha(alpha); break;
                    case JointType::T: j.set_alpha(alpha/2.0); break;
                    default: throw std::runtime_error("Unknown joint type");
                }
            }
        }
    }

    // Call f(joint_idx, lower_bound, upper_bound) for each joint with an angle
    // bound.
    template<class F>
    void visitAngleBounds(const F &f) const {
        const size_t nj = numJoints();
        for (size_t ji = 0; ji < nj; ++ji) {
            JointType t = joint(ji).jointType();
            if      (t == JointType::Rigid) continue;
            else if (t == JointType::X) { f(ji, 0.0, M_PI);     }
            else if (t == JointType::T) { f(ji, 0.0, M_PI / 2); }
            else throw std::runtime_error("Unknown joint type");
        }
    }

    // Compute the minimum over all joints of the joint opening angle.
    Real_ getMinJointAngle() const {
        Real_ result = safe_numeric_limits<Real>::max();
        for (const auto &j : m_joints) {
            if (j.jointType() != JointType::Rigid) result = std::min(result, j.alpha());
        }
        return result;
    }

    // The shortest rest-length of any rod in the linkage defines the characteristic lengthscale of this network.
    // (For purposes of determining reasonable descent velocites).
    Real_ characteristicLength() const {
        Real_ minLen = std::numeric_limits<float>::max();
        for (auto &s : m_segments) minLen = std::min(minLen, s.rod.characteristicLength());
        return minLen;
    }

    // Indices of all length quantity variables; we will want bound constraints
    // to keep these strictly positive.
    std::vector<size_t> lengthVars(bool variableDesignParameters = false) const {
        std::vector<size_t> result;
        // The two variables for each joint...
        for (size_t ji = 0; ji < numJoints(); ++ji) {
            for (size_t si = 0; si < joint(ji).valence(); ++si) {
                result.push_back(m_dofOffsetForJoint[ji] + joint(ji).numBaseDoF() + si);
            }
        }
        // ... and all the rest lengths, if requested
        if (variableDesignParameters && m_umbrella_dPC.restLen) {
            const size_t nfrl = numFreeRestLengths(),
                         frlo = freeRestLenOffset();
            for (size_t i = 0; i < nfrl; ++i)
                result.push_back(frlo + i);
            const size_t njrl = numJointRestLengths(),
                         jrlo = jointRestLenOffset();
            for (size_t i = 0; i < njrl; ++i)
                result.push_back(jrlo + i);        }
        return result;
    }

    // Get the index of the joint closest of the center of the structure.
    // This is usually a good choice for the joint used to constrain the
    // structures global rigid motion/drive it open.
    size_t centralJoint() const {
        Pt3 center(Pt3::Zero());
        for (const auto &j : m_joints) { center += j.pos(); }
        center /= numJoints();
        Real_ closestDistSq = safe_numeric_limits<Real_>::max();
        size_t closestIdx = 0;
        for (size_t ji = 0; ji < numJoints(); ++ji) {
            Real_ distSq = (m_joints[ji].pos() - center).squaredNorm();
            if (distSq < closestDistSq) {
                closestDistSq = distSq;
                closestIdx = ji;
            }
        }
        return closestIdx;
    }

    std::vector<size_t> rigidJoints() const {
        std::vector<size_t> result;
        result.reserve(m_numRigidJoints);
        for (size_t i = 0; i < numJoints(); ++i) {
            if (m_joints[i].jointType() == JointType::Rigid) result.push_back(i);
        }
        return result;
    }

    // Indices of the degrees of freedom controlling the joint center positions.
    std::vector<size_t> jointPositionDoFIndices() const {
        std::vector<size_t> result;
        const size_t nj = numJoints();
        result.reserve(3 * nj);
        for (size_t i = 0; i < nj; ++i) {
            size_t o = m_dofOffsetForJoint[i];
            result.push_back(o + 0);
            result.push_back(o + 1);
            result.push_back(o + 2);
        }
        return result;
    }

    /////////////
    // Repulsion
    ////////////

    Real_ getRepulsionEnergyWeight() const { return m_repulsionEnergyWeight; }
    void setRepulsionEnergyWeight(const Real_ val) { m_repulsionEnergyWeight = val; }

    DeploymentForceType getDeploymentForceType() const { return m_dftype; }
    void setDeploymentForceType(const DeploymentForceType dftype) { m_dftype = dftype; }

    AngleBoundEnforcement getAngleBoundEnforcement() const { return m_angleBoundEnforcement; }
    void setAngleBoundEnforcement(const AngleBoundEnforcement abe) { m_angleBoundEnforcement = abe; }

    Real_ getUniformDeploymentEnergyWeight() const { return m_uniformDeploymentEnergyWeight; }
    void setUniformDeploymentEnergyWeight(const Real_ val) {
        m_uniformDeploymentEnergyWeight = val;
        m_deploymentEnergyWeight.resize(numUmbrellas());
        for (int i = 0; i < m_deploymentEnergyWeight.size(); ++i)
            m_deploymentEnergyWeight[i] = val;
    }

    VecX getDeploymentEnergyWeight() const { return m_deploymentEnergyWeight; }
    void setDeploymentEnergyWeight(const VecX val) { 
        if (val.size() != numUmbrellas()) throw std::runtime_error("Input deployment weight doesn't match the number of umbrellas!");
        m_deploymentEnergyWeight = val; 
    }

    Real_ getTargetDeploymentHeight() const { return m_targetDeploymentHeight[0]; }
    void setTargetDeploymentHeight(const Real_ val) { m_targetDeploymentHeight.setConstant(val); }

    VecX getTargetDeploymentHeightVector() const { return m_targetDeploymentHeight; }
    void setTargetDeploymentHeightVector(const VecX val) { m_targetDeploymentHeight = val; }

    VecX topJointPositions() const {
        VecX result(3 * numUmbrellas());
        for (size_t ji = 0; ji < numUmbrellas(); ++ji)
            result.template segment<3>(3 * ji) = m_joints[m_umbrella_to_top_bottom_joint_map[ji][0]].pos();
        return result;
    }
    VecX bottomJointPositions() const {
        VecX result(3 * numUmbrellas());
        for (size_t ji = 0; ji < numUmbrellas(); ++ji)
            result.template segment<3>(3 * ji) = m_joints[m_umbrella_to_top_bottom_joint_map[ji][1]].pos();
        return result;
    }

    // top index at 0, bottom at 1;
    size_t getUmbrellaCenterJi(size_t ui, size_t ti) const { 
        if (ui >= m_umbrella_to_top_bottom_joint_map.size()) throw std::runtime_error("Index out of range! (getUmbrellaCenterJi)!");
        return m_umbrella_to_top_bottom_joint_map[ui][ti]; } 

    VecX getUmbrellaHeights() const {
        const size_t nu = numUmbrellas();
        VecX result(nu);
        for (size_t ui = 0; ui < nu; ++ui) {
            result[ui] = (m_joints[getUmbrellaCenterJi(ui, 0)].pos() -
                          m_joints[getUmbrellaCenterJi(ui, 1)].pos()).norm();
        }
        return result;
    }

    void setOppositeCenter() {
        for (size_t ui = 0; ui < numUmbrellas(); ++ui) {
            size_t top_ji = getUmbrellaCenterJi(ui, 0);
            size_t bot_ji = getUmbrellaCenterJi(ui, 1);
            m_joints[top_ji].setOppositeCenter(bot_ji);
            m_joints[bot_ji].setOppositeCenter(top_ji);
        }
    }

    size_t getArmIndexAt(size_t segment_index) const { return m_armIndexForSegment[segment_index]; }
    size_t getSegmentIndexForArmAt(size_t arm_index) const { return m_segmentIndexForArm[arm_index]; }

    /////////////////////////
    // Target Surface Fitting
    /////////////////////////
    VecX XJointPositions() const {
        VecX result(3 * numXJoints());
        for (size_t ji = 0; ji < numXJoints(); ++ji)
            result.template segment<3>(3 * ji) = m_joints[m_X_joint_indices[ji]].pos();
        return result;
    }
    VecX XJointTgtPositions() const {
        VecX result(3 * numXJoints());
        for (size_t ji = 0; ji < numXJoints(); ++ji)
            result.template segment<3>(3 * ji) = m_X_joint_tgt_pos[ji];
        return result;
    }

    VecX UmbrellaMidTgtPositions() const {
        VecX result(3 * numUmbrellas());
        for (size_t ui = 0; ui < numUmbrellas(); ++ui)
            result.template segment<3>(3 * ui) = m_umbrella_tgt_pos[ui];
        return result;
    }

    std::vector<bool> IsQueryPtBoundary() const {
        std::vector<bool> result(numXJoints() + numUmbrellas(), false);
        for (size_t ji = 0; ji < numXJoints(); ++ji)
            if (m_joints[m_X_joint_indices[ji]].valence() == 2) result[ji] = true;
        return result;
    }

    const Joint &X_joint(size_t i) const { return m_joints[m_X_joint_indices[i]]; }
          Joint &X_joint(size_t i)       { return m_joints[m_X_joint_indices[i]]; }

    size_t get_X_joint_indice_at(size_t i) const { return m_X_joint_indices[i]; }

    // Get a list of all the centerline positions in the network.
    std::vector<Pt3> deformedPoints() const {
        std::vector<Pt3> result;
        for (const auto &s : m_segments) {
            const auto &dp = s.rod.deformedPoints();
            result.insert(result.end(), dp.begin(), dp.end());
        }
        return result;
    }

    VecX centerLinePositions() const {
        VecX result(3 * numCenterlinePos());
        size_t offset = 0;
        for (const auto &s : m_segments) {
            const auto &dof = s.rod.getDoFs();
            size_t range = 3 * s.numFreeVertices();
            result.segment(offset, range) = dof.segment(6 * s.hasStartJoint(), range);
            offset += range;
        }
        return result;
    }

    bool hasTargetSurface() const { return m_target_surface_fitter != nullptr; }
    std::shared_ptr<TargetSurfaceFitter> getTargetSurface() const { return m_target_surface_fitter; }

    Real get_l0() const { return m_l0; }
    Real get_E0() const { return m_E0; }
    
    Real getAttractionWeight() const { return m_attraction_weight; }
    void setAttractionWeight(Real input_weight) { m_attraction_weight = input_weight; }

    LinearActuators<UmbrellaMesh_T> &getLinearActuator() { return m_linearActuator; }

    /////////////////////////
    // Four Parameters
    /////////////////////////
    bool isSegmentTop(size_t si) const {
        size_t sJoint = segment(si).startJoint, eJoint = segment(si).endJoint;
        size_t tJoint = (sJoint != NONE && joint(sJoint).jointType() == UmbrellaMesh::JointType::X) ? eJoint : sJoint;
        return (tJoint != NONE && joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Top);
    }

    size_t getArmUID(size_t si) const {
        size_t sJoint = segment(si).startJoint, eJoint = segment(si).endJoint;
        size_t tJoint = (joint(sJoint).jointType() == UmbrellaMesh::JointType::X) ? eJoint : sJoint;
        return joint(tJoint).umbrellaID()[0]; // Fetch the current umbrella id of t-joint. Index 1 would have the neighbor ID.
    }

    size_t getMirrorArm(size_t si) const {
        size_t sJoint = segment(si).startJoint, eJoint = segment(si).endJoint;
        size_t xJoint = (joint(sJoint).jointType() == UmbrellaMesh::JointType::X) ? sJoint : eJoint;
        return getArmIndexAt(joint(xJoint).getMirrorArm(si));
    }

    size_t numBottomArmSegments() const {
        size_t num_bottom_arms = numArmSegments();
        for (size_t ai = 0; ai < numArmSegments(); ++ai) {
            if (isSegmentTop(getSegmentIndexForArmAt(ai))) num_bottom_arms -= 1;
        }
        return num_bottom_arms;
    }
        
    virtual ~UmbrellaMesh_T() { }

	struct Joint {
        struct TerminalEdgeInputData {
            TerminalEdgeInputData(size_t i, Real_ l, const Vec3 &t, const Vec3 &n, const Vec3 &p, bool isA, bool start)
                : si(i), len(l), world_t(t), world_normal(n), world_p(p), is_A(isA), isStart(start) { }
            size_t si;          // Segment containing this terminal edge
            Real_ len;          // Initial length of the terminal edge.
            Vec3 world_t,       // Edge tangent in world coordinates
                 world_normal,  // Edge normal (d2 vector) in world coordinates
                 world_p;       // Edge midpoint *offset* in world coordinates
            bool is_A;          // Is this edge is attached to the "A" or "B" rigid body?
            bool isStart;       // Is this terminal edge at the start or end of the segment it belongs to?
        };

        // Converting constructor from a different floating point type.
        // Unfortunately, we can't just template this on the floating point type
        // of the surrounding UmbrellaMesh_T struct, since then the compiler isn't
        // able to deduce the template parameter...
        template<typename Joint2>
        Joint(const Joint2 &j)
            : m_umbrella_mesh(nullptr) { m_setState<decltype(j.alpha())>(j.getState()); }


        // Sadly cannot be deduced from getState()'s return (with auto)
        using SerializedState = std::tuple<Pt3, Vec3, Real_, VecX, 
                                           Vec3, Vec3,
                                           Mat3X, Mat3X, Mat3X,
                                           Mat3X, Mat3X, Mat3X,
                                           std::vector<size_t>, std::vector<bool>, size_t, size_t,
                                           size_t, std::vector<size_t>, std::vector<size_t>, 
                                           JointType, JointPosType, std::vector<size_t>, size_t>;
        // Needed for the pickling to work.
        Joint(const SerializedState &state) : m_umbrella_mesh(nullptr) { m_setState<Real_>(state); }

        // Get full state of this Joint, e.g. for serialization
        SerializedState getState() const {
            return std::make_tuple(m_pos, m_omega, m_alpha, m_len, 
                                   m_ghost_source_t, m_ghost_source_normal, 
                                   m_source_t, m_source_normal, m_source_p, 
                                   m_input_t, m_input_normal, m_input_p,
                                   m_segments, m_isStart, num_A_segments, num_B_segments, 
                                   m_numArmSegments, m_armIndexForSegment, m_segmentIndexForArm,
                                   m_jointType, m_jointPosType, m_umbrella_ID, m_opposite_center);
        }

        // Set the cached state, e.g., for serialization (use with care!)
        template<typename Real2_>
        void m_setState(const typename UmbrellaMesh_T<Real2_>::Joint::SerializedState &state) {
            m_pos                 = std::get< 0>(state);
            m_omega               = std::get< 1>(state);
            m_alpha               = std::get< 2>(state);
            m_len                 = std::get< 3>(state);
            m_ghost_source_t      = std::get< 4>(state);
            m_ghost_source_normal = std::get< 5>(state);
            m_source_t            = std::get< 6>(state);
            m_source_normal       = std::get< 7>(state);
            m_source_p            = std::get< 8>(state);
            m_input_t             = std::get< 9>(state);
            m_input_normal        = std::get<10>(state);
            m_input_p             = std::get<11>(state);
            m_segments            = std::get<12>(state);
            m_isStart             = std::get<13>(state);
            num_A_segments        = std::get<14>(state);
            num_B_segments        = std::get<15>(state);
            m_numArmSegments      = std::get<16>(state);
            m_armIndexForSegment  = std::get<17>(state);
            m_segmentIndexForArm  = std::get<18>(state);
            m_jointType           = JointType(std::get<19>(state));
            m_jointPosType        = JointPosType(std::get<20>(state));
            m_umbrella_ID         = std::get<21>(state);
            m_opposite_center     = std::get<22>(state);

            m_update();
        }

        Joint(UmbrellaMesh_T *um, const Pt3 &p, const Real_ alpha,
              const Vec3 &normal, const Vec3 &bisector,
              const std::vector<TerminalEdgeInputData> &inputTerminalEdges,
              const JointType &jointType,
              const std::vector<size_t> &umbrella_ID);

        size_t numBaseDoF() const { return 7; }
        size_t valence() const { return num_A_segments + num_B_segments; }
        size_t numArms() const { return m_numArmSegments; }
        size_t  numDoF() const { return numBaseDoF() + valence(); }
        const Vec3 &  pos() const { return m_pos; }
        const Vec3 &omega() const { return m_omega; }
        Real_       alpha() const { return m_alpha; }
        Vec3       normal() const { return ropt::rotated_vector(m_omega, m_ghost_source_normal); }
        Vec3 ghost_normal() const { return ropt::rotated_vector(m_omega, m_ghost_source_normal); }
        Real_       len(size_t idx) const { return m_len[idx]; }
        const JointType & jointType() const {return m_jointType;}
        const JointPosType & jointPosType() const {return m_jointPosType;}
        const std::vector<size_t> & umbrellaID() const {return m_umbrella_ID;}
        // Set the joint parameters from a collection of global variables (DoFs)
        template<class Derived>
        void setParameters(const Eigen::DenseBase<Derived> &vars);

        // Change the rotation parametrization to be the tangent space of SO(3) at the current rotation.
        // Note: this changes omega (and consequently the linkage DoF values).
        void updateParametrization() {
            m_ghost_source_t      = ropt::rotated_vector(m_omega, m_ghost_source_t);
            m_ghost_source_normal = ropt::rotated_vector(m_omega, m_ghost_source_normal);
            m_update_source_info();
            m_omega.setZero();
        }

        void updateUmbrellaMeshPointer(UmbrellaMesh_T *ptr) { m_umbrella_mesh = ptr; }

        const UmbrellaMesh_T &umbrellaMesh() const { assert(m_umbrella_mesh != nullptr); return *m_umbrella_mesh; }
              UmbrellaMesh_T &umbrellaMesh()       { assert(m_umbrella_mesh != nullptr); return *m_umbrella_mesh; }

        // Extract the joint parameters, storing them in a collection of global variables (DoFs)
        template<class Derived>
        void getParameters(Eigen::DenseBase<Derived> &vars) const;

        // Update the deformed points/material frames of the incident rod edges,
        // as stored in the full network's "points" and "thetas" arrays.
        void applyConfiguration(const std::vector<RodSegment>   &rodSegments,
                                std::vector<std::vector<Pt3>>   &networkPoints,
                                std::vector<std::vector<Real_>> &networkThetas,
                                bool spatialCoherence = false) const;

        // Determine the sensitivity of the incident terminal edge vector of segment "si"
        // to changes to eA and eB. This is represented by the pair of scalars
        // (s_jA, s_jB) where, e.g., s_jA is:
        //      0 if edge vector A doesn't control terminal edge j,
        //      1 if edge vector A gives terminal edge j's vector directly,
        //     -1 if edge vector A gives the negative of terminal edge j's vector
        std::tuple<bool, bool, size_t> terminalEdgeIdentification(size_t si) const;
        size_t getSegmentAt(const size_t localSegmentIndex) const {
            if (localSegmentIndex >= num_A_segments + num_B_segments) throw std::runtime_error("localSegmentIndex out of bounds");
            return m_segments[localSegmentIndex];
        }

        bool getIsStartAt(const size_t localSegmentIndex) const {
            if (localSegmentIndex >= num_A_segments + num_B_segments) throw std::runtime_error("localSegmentIndex out of bounds");
            return m_isStart[localSegmentIndex];
        }

        SegmentType getSegmentTypeAt(const size_t localSegmentIndex) const {
            return umbrellaMesh().segment(getSegmentAt(localSegmentIndex)).segmentType();
        }

        // Offset for the segment length in the full parameters vector starting from the first segment length.
        size_t len_offset(const size_t si) const {
            size_t localSegmentIndex;
            bool is_A, isStart;
            std::tie(is_A, isStart, localSegmentIndex) = terminalEdgeIdentification(si);
            return localSegmentIndex;
        }

        // Offset for the segment rest length in the design parameters vector starting from the first segment rest length.
        size_t arm_offset(const size_t localSegmentIndex) const { return m_armIndexForSegment[localSegmentIndex]; }

        // Offset for the segment rest length in the design parameters vector starting from the first segment rest length.
        size_t arm_offset_for_global_segment(const size_t si) const {
            size_t localSegmentIndex;
            bool is_A, isStart;
            std::tie(is_A, isStart, localSegmentIndex) = terminalEdgeIdentification(si);
            return arm_offset(localSegmentIndex);
        }

        size_t getSegmentIndexForArmAt(size_t arm_index) const { return m_segmentIndexForArm[arm_index]; }

        // Rod A,B's edge tangents (before rotation by omega).
        // Ghost edge A is rotated by + alpha / 2 around the rotation axis (joint normal).
        // Ghost edge B is rotated by - alpha / 2 around the rotation axis (joint normal).
        const Vec3 &ghost_source_t() const { return m_ghost_source_t; }
        const Vec3 &ghost_source_normal() const { return m_ghost_source_normal; }
        const Vec3 &      source_normal() const { return m_ghost_source_normal; }
        Vec3 ghost_source_nxb() const { 
            Vec3 nxb = (m_ghost_source_normal.cross(m_ghost_source_t)); 
            if (nxb.norm() < 1e-8) std::cerr<<"Input source vector is parallel to input normal"<<std::endl;
            return nxb;
        }
        Mat3 get_ghost_source_frame() const {
            Mat3 ghost_source_config;
            ghost_source_config.col(0) = ghost_source_t();
            ghost_source_config.col(1) = ghost_source_nxb(); 
            ghost_source_config.col(2) = ghost_source_normal();
            return ghost_source_config;
        }
        Vec3 ghost_source_t_A()   const { return ropt::rotated_vector((-0.5 * m_alpha) * m_ghost_source_normal, m_ghost_source_t); }
        Vec3 ghost_source_nxt_A() const { return ropt::rotated_vector((-0.5 * m_alpha) * m_ghost_source_normal, ghost_source_nxb()); }
        Vec3 ghost_source_t_B()   const { return ropt::rotated_vector(( 0.5 * m_alpha) * m_ghost_source_normal, m_ghost_source_t); }
        Vec3 ghost_source_nxt_B() const { return ropt::rotated_vector(( 0.5 * m_alpha) * m_ghost_source_normal, ghost_source_nxb()); }

        // orthogonal frame formed by the ghost edge A and the ghost normal; and the frame formed by the ghost edge B and the ghost normal; R_0R_alpha.
        Mat3 get_ghost_source_frame_A() const {
            Mat3 ghost_source_config_A;
            ghost_source_config_A.col(0) = ghost_source_t_A();
            ghost_source_config_A.col(1) = ghost_source_nxt_A(); 
            ghost_source_config_A.col(2) = ghost_source_normal();
            return ghost_source_config_A;
        }
        Mat3 get_ghost_source_frame_B() const {
            Mat3 ghost_source_config_B;
            ghost_source_config_B.col(0) = ghost_source_t_B();
            ghost_source_config_B.col(1) = ghost_source_nxt_B(); 
            ghost_source_config_B.col(2) = ghost_source_normal();
            return ghost_source_config_B;
        }

        // Used for computing the derivatives against the alpha variable.
        Mat3 get_minus_alpha_sensitivity() const {
            // d R(-alpha/2) / d alpha
            Mat3 sensitivity;
            Real_ half_alpha = 0.5 * m_alpha;
            sensitivity << - sin(half_alpha),   cos(half_alpha), 0,
                           - cos(half_alpha), - sin(half_alpha), 0,
                                           0,                 0, 0;
            return 0.5 * sensitivity;
        }
        
        Mat3 get_plus_alpha_sensitivity() const {
            // d R( alpha/2) / d alpha
            Mat3 sensitivity;
            Real_ half_alpha = 0.5 * m_alpha;
            sensitivity << - sin(half_alpha), - cos(half_alpha), 0,
                             cos(half_alpha), - sin(half_alpha), 0,
                                           0,                 0, 0;
            return 0.5 * sensitivity;
        }

        Mat3 get_ghost_alpha_sensitivity_frame_A() const {
            return get_ghost_source_frame() * get_minus_alpha_sensitivity();
        }

        Mat3 get_ghost_alpha_sensitivity_frame_B() const {
            return get_ghost_source_frame() *  get_plus_alpha_sensitivity();
        }

        // Return a matrix for R''(theta)
        Mat3 angle_hessian(Real_ theta) const {
            Mat3 hessian;
            hessian << -cos(theta),  sin(theta), 0,
                       -sin(theta), -cos(theta), 0, 
                                 0,           0, 0;
            return hessian;
        }

        Mat3 get_plus_alpha_hessian() const {
            return 0.25 * angle_hessian(0.5 * m_alpha);
        }

        Mat3 get_minus_alpha_hessian() const {
            return 0.25 * angle_hessian(-0.5 * m_alpha);
        }

        Mat3 get_ghost_alpha_hessian_frame_A() const {
            return get_ghost_source_frame() * get_minus_alpha_hessian();
        }

        Mat3 get_ghost_alpha_hessian_frame_B() const {
            return get_ghost_source_frame() *  get_plus_alpha_hessian();
        }
        ////////////////////////////////////////////////////////////////

        // Access source info.
        Vec3 source_t(size_t idx) const { return m_source_t.col(idx); }
        Vec3 source_normal(size_t idx) const { return m_source_normal.col(idx); }
        Vec3 source_p(size_t idx) const { return m_source_p.col(idx); }
        
        ///////////////////////
        // Normal vector for each segment.
        Vec3 normal(size_t idx) const { return m_segment_normal.col(idx); }
        
        // Access input info.
        Vec3 input_t(size_t idx) const { return m_input_t.col(idx); }
        Vec3 input_normal(size_t idx) const { return m_input_normal.col(idx); }
        Vec3 input_p(size_t idx) const { return m_input_p.col(idx); }
        
        ///////////////////////
        void set_pos  (const Vec3 &v)     { m_pos = v; }
        void set_omega(const Vec3 &omega) { m_omega = omega; m_update(); }
        void set_alpha(Real_ alpha)       { m_alpha = alpha; m_update(); }

        // Get the rest lengths for the edges this joint controls.
        VecX getRestLengths() const {
            const auto &um = umbrellaMesh();
            auto getLen = [&](size_t sidx, bool isStart) {
                const auto &r = um.segment(sidx).rod;
                return r.restLengthForEdge(isStart ? 0 : (r.numEdges() - 1));
            };
            VecX result(numArms());
            size_t back = 0;
            for (size_t idx = 0; idx < valence(); ++idx) {
                if (getSegmentTypeAt(idx) == SegmentType::Arm)
                    result[back++] = getLen(getSegmentAt(idx), getIsStartAt(idx));
            }
            return result;
        }

        // Set the rest lengths for the edges this joint controls (one for the rods of segment A, one for B)
        void setRestLengths(const VecX &rlens) {
            auto &um = umbrellaMesh();
            auto setLen = [&](size_t sidx, bool isStart, Real_ val) {
                if (sidx == NONE) throw std::runtime_error("The joint shouldn't store NONE segment!");
                auto &r = um.segment(sidx).rod;
                r.restLengthForEdge(isStart ? 0 : (r.numEdges() - 1)) = val;
            };
            size_t back = 0;
            for (size_t idx = 0; idx < valence(); ++ idx) {
                if (getSegmentTypeAt(idx) == SegmentType::Arm)
                    setLen(getSegmentAt(idx), getIsStartAt(idx), rlens[back++]);
            }
        }

        size_t getOppositeCenter() const { return m_opposite_center; }
        void   setOppositeCenter(size_t ji) { m_opposite_center = ji; }

        // Four parameters.
        size_t getMirrorArm(size_t si) const {
            auto &um = umbrellaMesh();
            size_t uid = um.getArmUID(si);
            for (size_t idx = 0; idx < valence(); ++idx) {
                if (getSegmentTypeAt(idx) == SegmentType::Arm && getSegmentAt(idx) != si) {
                    if (um.getArmUID(getSegmentAt(idx)) == uid) return getSegmentAt(idx);
                }
            }
            throw std::runtime_error("This segment has no mirror in the joint!");
        }

        // Hessian Related.
        
        // Call "visitor(idx)" for each global independent vertex/theta of
        // freedom index "idx" influenced by the joint's variable "var"
        // (i.e. that appears in the joint var's column of the Hessian).
        // Note: global degrees of freedom are *not* visited in order.
        // restLenVar: whether "var" selects an ordinary joint variable or
        // a joint rest length variable.
        template<class F>
        void visitInfluencedSegmentVars(const size_t var, F &visitor, bool restLenVar = false) const;
        ///////////////////
        
    	protected:
	    	// Pointer to the umbrella mesh containing this joint; needed for accessing
	        // the segments controlled by this joint.
	        // This is a pointer rather than a reference to enable the RodLinkage's
	        // copy construction (though care should be taken when copying joints).
	        UmbrellaMesh_T *m_umbrella_mesh;
			
			// Joint parameters:
	        Pt3   m_pos;             // Position of the joint (origin of the rotation axis).
	        Vec3  m_omega;           // The axis-angle representation of the joint's rotation (in tangent space of SO(3))
	        Real_ m_alpha;           // The angle between the two ghost edges. This is defined as the opening angle of the joint. 
            VecX m_len; // The length of each incident edge attached to ghost edges (first A then B).

            // Descriptive quantities:
            size_t num_A_segments = 0, num_B_segments = 0;

	        // Derived quantities:
            Vec3 m_ghost_A, m_ghost_B;  // The incident ghost edge vector A/B. Use for visualization and debugging only.
            Mat3X m_e;  // The edge direction vector for each rod segment attached to ghost edges. The dimension is num_segments x 3 and num_B_segments x 3. Denoted as t_1 in the notes. The length of the vectors is specified by m_len.
            Mat3X m_segment_normal; // The normal of the rod segments attached to the ghost edges.
            Mat3X m_p;  // The edge midpoint offset vector for each rod segment attached to ghost edges. The dimension is num_segments x 3 and num_B_segments x 3. The absolute edge midpoint position is specified as m_pos + m_p.

            // The "reference" rotation around which the joint's rotation is
            // parametrized is given by
            // (m_ghost_source_t | m_ghost_source_normal x m_ghost_source_t | m_ghost_source_normal).
            // The current tangent and normal are the rotation of m_ghost_source_t and
            // m_ghost_source_normal by the rotation described by axis/angle m_omega.
            Vec3 m_ghost_source_t, m_ghost_source_normal;

            // The input edge tangents, normals and midpoint positions of segments attached to ghost edges in the joint's local frame. 
            Mat3X m_input_t, m_input_normal, m_input_p;

            // The source edge tangents, normals and midpoint positions of segments attached to ghost edges in the joint's local frame. 
            Mat3X m_source_t, m_source_normal, m_source_p;

            // Connectivity information
            std::vector<size_t> m_segments; // The list of segments that attached to the ghost edges.
            std::vector<bool> m_isStart;  // Whether this joint is at the start (or end) of the incident rod segments.

            // Joint Type
            JointType m_jointType;

            // Joint Pos Type
            JointPosType m_jointPosType;

            size_t m_numArmSegments = 0;
            std::vector<size_t> m_armIndexForSegment;
            std::vector<size_t> m_segmentIndexForArm;
            // The ID of the umbrella that has this joint. 
            std::vector<size_t> m_umbrella_ID;
            size_t m_opposite_center = NONE;
            // ToDo: Serialization.

            // Update the source frame for each attached segment using the new alpha.
            void m_update_source_info();
            // Update cached edge vectors/normals; to be called whenever the parameters change.
            void m_update();
	};

    // RodSegment parameters are the underlying elastic rod's centerline
    // positions and material frame angles (thetas). The first and last edge of
    // the rod only have degrees of freedom if that rod end is not part of a
    // joint.
    struct RodSegment {
        size_t startJoint = NONE, endJoint = NONE;
        bool hasStartContinuation = false, hasEndContinuation = false; // Indicate whether the segment is elastically connecting with another segment through the joints.

        Rod rod;

        // Converting constructor from a different floating point type.
        // Unfortunately, we can't just template this on the floating point type
        // of the surrounding UmbrellaMesh_T struct, since then the compiler isn't
        // able to deduce the template parameter...
        template<typename RodSegment2>
        RodSegment(const RodSegment2 &s)
            : startJoint(s.startJoint), endJoint(s.endJoint), rod(s.rod), m_segmentType(s.segmentType()) { }

        // Construct a rod segment with endpoint pivots at startPt and endPt.
        // Because the pivots occur at an edge midpoint, the rod will extend
        // half an edgelength past the start and end points.
        // TODO: change nsubdiv to an edge length parameter!
        RodSegment(const Pt3 &startPt, const Pt3 &endPt, size_t nsubdiv, SegmentType sType = SegmentType::Arm);

        RodSegment(size_t nsubdiv, std::function<Pt3_T<Real_>(Real_)> edge_callback, const Real_ start_len, const Real_ end_len);

        RodSegment(size_t startJoint, size_t endJoint, Rod &&rod, SegmentType sType = SegmentType::Arm) :
            startJoint(startJoint), endJoint(endJoint), rod(std::move(rod)), m_segmentType(sType) { }

        // Read the rod parameters from a collection of global variables (DoFs),
        // storing them in "points" and "thetas" arrays. Only the entries of "points"
        // and "thetas" that are controlled by this rod segment's parameters are
        // altered (terminal edge quantities controlled by a joint are left unchanged).
        template<class Derived>
        void unpackParameters(const Eigen::DenseBase<Derived> &vars,
                              std::vector<Pt3>   &points,
                              std::vector<Real_> &thetas) const;

        void initializeCenterlineOffset(std::vector<Pt3  > &points) const;

        // Set the rod parameters from a collection of global variables (DoFs)
        template<class Derived>
        void setParameters(const Eigen::DenseBase<Derived> &vars);

        // Extract the rod parameters, storing them in a collection of global variables (DoFs)
        template<class Derived>
        void getParameters(Eigen::DenseBase<Derived> &vars) const;

        bool hasStartJoint() const { return startJoint != NONE; }
        bool hasEndJoint()   const { return   endJoint != NONE; }
        size_t numJoints()   const { return hasStartJoint() + hasEndJoint(); }
        // Access start/end joint indices by label 0/1.
        size_t joint(size_t i) const { if (i == 0) return startJoint;
                                       if (i == 1) return endJoint;
                                       throw std::runtime_error("Out of bounds.");
        }
        size_t localJointIndex(const size_t ji) const {
            if (ji == startJoint) return 0;
            if (ji == endJoint  ) return 1;
            throw std::runtime_error("Joint " + std::to_string(ji) + " is not incident this segment.");
        }

        // Determine the number of degrees of freedom belonging to this segment's rod
        // (after joint constraints have been applied).
        size_t numDoF() const {
            // Joints (if present) each determine the position of two vertices
            // and the material axis of one edge.
            return rod.numDoF() - numJoints() * (2 * 3 + 1);
        }

        // Number of internal/free end vertices and edges
        size_t numFreeVertices() const { return rod.numVertices() - 2 * (numJoints()); }
        size_t numFreeEdges()    const { return rod.numEdges()    -      numJoints() ; }

        // Determine the number of positional degrees of freedom belonging to this
        // segment's rod after joint constraints have been applied. This effectively
        // gives us the offset into the segment's reduced DoFs of the first
        // material frame variable.
        size_t numPosDoF() const { return 3 * (rod.numVertices() - 2 * hasStartJoint() - 2 * hasEndJoint()); }

        // Number of degrees of freedom in the underlying rod (i.e. before
        // joint constraints have been applied).
        size_t fullDoF() const { return rod.numDoF(); }
        // This is actually pretty good.

        // Update the unconstrained thetas (internal edges + free ends) to
        // minimize twisting energy
        void setMinimalTwistThetas(bool verbose = false);

        const SegmentType & segmentType() const {return m_segmentType;}
        const ArmSegmentPosType & armSegmentPosType() const {return m_armSegmentPosType;}
        void setArmSegmentPosType(const ArmSegmentPosType type) {m_armSegmentPosType = type;}
        protected:
            // Segment Type
            SegmentType m_segmentType;
            // Segment Pos Type
            ArmSegmentPosType m_armSegmentPosType = ArmSegmentPosType::Null;
    };
    ////////////////////////////////////////////////////////////////////////////
    // Expose TerminalEdgeSensitivity for debugging
    ////////////////////////////////////////////////////////////////////////////
    enum class TerminalEdge : int { Start = 0, End = 1 };
    const UmbrellaMeshTerminalEdgeSensitivity<Real_> &getTerminalEdgeSensitivity(size_t si, TerminalEdge which, bool updatedSource, bool evalHessian);
    const UmbrellaMeshTerminalEdgeSensitivity<Real_> &getTerminalEdgeSensitivity(size_t si, TerminalEdge which, bool updatedSource, const VecX &delta_params);

    std::string mangledName() const { return "UmbrellaMesh<" + autodiffOrNotString<Real_>() + ">"; }


    ////////////////////////////////////////////////////////////////////////////
    // Visualization
    ////////////////////////////////////////////////////////////////////////////
    void coloredVisualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                                      std::vector<MeshIO::IOElement> &quads,
                                      const bool averagedMaterialFrames,
                                      const bool averagedCrossSections,
                                      Eigen::VectorXd *height = nullptr) const;

    void visualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                               std::vector<MeshIO::IOElement> &quads,
                               const bool averagedMaterialFrames = false,
                               const bool averagedCrossSections = false) const {
        coloredVisualizationGeometry(vertices, quads, averagedMaterialFrames, averagedCrossSections);
    }

    void saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames = false) const;


    enum class ScalarFieldType { UNKNOWN, PER_VERTEX, PER_EDGE };

    template<typename Getter>
    std::vector<Eigen::VectorXd> collectRodScalarFields(const Getter &getter) const {
        std::vector<Eigen::VectorXd> result;
        result.reserve(numSegments());
        auto type = ScalarFieldType::UNKNOWN;
        for (size_t si = 0; si < numSegments(); ++si) {
            const auto &r = segment(si).rod;
            result.push_back(getter(r));
            ScalarFieldType currType;
            if      (size_t(result.back().size()) == r.numVertices()) currType = ScalarFieldType::PER_VERTEX;
            else if (size_t(result.back().size()) == r.numEdges()) currType = ScalarFieldType::PER_EDGE;
            else throw std::runtime_error("Unrecognized rod scalar field type");
            if (type == ScalarFieldType::UNKNOWN) type = currType;
            if (type != currType) throw std::runtime_error("Rod scalar fields are of mixed types (illegal)");
        }

        // // Vertex-based scalar fields must be made continuous across joints.
        // // For example, bending and twisting stresses are zero at the segment
        // // end vertices and we should copy the values from the coinciding
        // // vertices of the continuation segments.
        // if (type == ScalarFieldType::PER_VERTEX) {
        //     for (size_t su = 0; su < numSegments(); ++su) {
        //         for (size_t lji = 0; lji < 2; ++lji) {
        //             size_t ji = segment(su).joint(lji);
        //             if (ji == NONE) continue;
        //             size_t sv, is_start_sv;
        //             std::tie(sv, is_start_sv) = joint(ji).continuationSegmentInfo(su);
        //             if (sv == NONE) continue;
        //             const Eigen::VectorXd &src = result[sv];
        //             Eigen::VectorXd &dest = result[su];
        //             dest[lji == 0 ? 0 : dest.size() - 1] = src[is_start_sv ? 1 : src.size() - 2];
        //         }
        //     }
        // }

        return result;
    }

    std::vector<Eigen::VectorXd> sqrtBendingEnergies() const { return collectRodScalarFields([](const Rod &r) { return stripAutoDiff(r.energyBendPerVertex()).array().sqrt().eval(); }); }
    std::vector<Eigen::VectorXd>  stretchingStresses() const { return collectRodScalarFields([](const Rod &r) { return r.stretchingStresses(); }); }
    std::vector<Eigen::VectorXd>  maxBendingStresses() const { return collectRodScalarFields([](const Rod &r) { return r.maxBendingStresses(); }); }
    std::vector<Eigen::VectorXd>  minBendingStresses() const { return collectRodScalarFields([](const Rod &r) { return r.minBendingStresses(); }); }
    std::vector<Eigen::VectorXd>    twistingStresses() const { return collectRodScalarFields([](const Rod &r) { return r.  twistingStresses(); }); }
    
    
    std::vector<Eigen::VectorXd>    maxVonMisesStresses() const { return collectRodScalarFields([](const Rod &r) { return r.  maxStresses(CrossSectionStressAnalysis::StressType::VonMises); }); }

    Real_ surfaceStressLpNorm(CrossSectionStressAnalysis::StressType type, double p, bool takeRoot = true) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".surfaceStressLpNorm");
        Real_ result = summation_parallel([&](size_t si) {
                return m_segments[si].rod.surfaceStressLpNorm(type, p, /* takeRoot = */ false);
            }, numSegments());
        return takeRoot ? pow(result, Real_(1 / p)) : result;
    }

    VecX gradSurfaceStressLpNormPerEdgeRestLen(CrossSectionStressAnalysis::StressType type, double p, bool updatedSource, bool takeRoot = true) const;
    VecX gradSurfaceStressLpNorm              (CrossSectionStressAnalysis::StressType type, double p, bool updatedSource, bool takeRoot = true) const;

    // Note: the rotation parametrization should be updated before calling
    // this method!
    VecX rivetForces(UmbrellaEnergyType type = UmbrellaEnergyType::Elastic, EnergyType eType = EnergyType::Elastic, bool needTorque = true) const;

    Eigen::MatrixXd UmbrellaRivetNetForceAndTorques(UmbrellaEnergyType type = UmbrellaEnergyType::Elastic, EnergyType eType = EnergyType::Elastic) const;

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
    visualizationField(const std::vector<Derived> &perRodSegmentFields) const {
        if (perRodSegmentFields.size() != numSegments()) throw std::runtime_error("Invalid field size");
        using FieldStorage = Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>;
        std::vector<FieldStorage> perRodVisualizationFields;
        perRodVisualizationFields.reserve(perRodSegmentFields.size());
        int fullSize = 0;
        const int cols = perRodSegmentFields.at(0).cols();
        for (size_t si = 0; si < numSegments(); ++si) {
            if (cols != perRodSegmentFields[si].cols()) throw std::runtime_error("Mixed field types forbidden.");
            perRodVisualizationFields.push_back(segment(si).rod.visualizationField(perRodSegmentFields[si]));
            fullSize += perRodVisualizationFields.back().rows();
        }
        FieldStorage result(fullSize, cols);
        int offset = 0;
        for (const auto &vf : perRodVisualizationFields) {
            result.block(offset, 0, vf.rows(), cols) = vf;
            offset += vf.rows();
        }
        assert(offset == fullSize);
        return result;
    }

protected:
    template<class F>
    void m_assembleSegmentGradient(const F &gradientGetter, VecX_T<Real_> &g,
                                   bool updatedSource, bool variableDesignParameters, bool designParameterOnly) const;

    std::vector<Joint> m_joints;
    std::vector<RodSegment> m_segments;
    size_t m_numRigidJoints;
    size_t m_numArmSegments;
    size_t m_numURH;
    // Indices of X Joints.
    std::vector<size_t> m_X_joint_indices;
    // Tgt_pos of X Joints.
    std::vector<Pt3> m_X_joint_tgt_pos; 

    // Tgt_pos of Midpoints of top and bot plate centers.
    std::vector<Pt3> m_umbrella_tgt_pos; 

    // Material used to initialize rods (useful if they are recreated by
    // RodLinkage::set after the linkage's material has been configured).
    RodMaterial m_armMaterial, m_plateMaterial;

    DesignParameterConfig m_umbrella_dPC;
    // Offset in the full list of linkage DoFs of the DoFs for each segment/joint
    std::vector<size_t> m_dofOffsetForSegment,
                        m_dofOffsetForJoint,
                        m_dofOffsetForCenterlinePos, 
                        m_restLenDofOffsetForSegment,
                        m_designParameterDoFOffsetForJoint;

    void m_buildDoFOffsets();

    Real_ m_initMinRestLen = 0;

    SuiteSparseMatrix m_armRestLenToEdgeRestLenMapTranspose; // Non-autodiff! (The map is piecewise constant/nondifferentiable).


    void m_constructArmRestLenToEdgeRestLenMapTranspose();
    void m_setRestLengthsFromPARL();

    VecX m_perArmRestLen; // cached for ease so we don't have to reconstruct from the linkage's per-edge rest length.
    std::vector<size_t> m_armIndexForSegment;
    std::vector<size_t> m_segmentIndexForArm;

    VecX m_designParametersPARL;
    // Cache to avoid memory allocation in setDoFs
    std::vector<std::vector<Pt3  >> m_networkPoints;
    std::vector<std::vector<Real_>> m_networkThetas;
 
    Real m_E0  = 1;

    // Target Surface Attraction. 
    std::shared_ptr<TargetSurfaceFitter> m_target_surface_fitter;
    Real m_l0 = 1;
    // To change this weight, call scaleJointWeight.
    Real m_attraction_input_joint_weight = 0.001;
    Real m_attraction_weight = 0;
    
    // Deployment
    Real_ m_uniformDeploymentEnergyWeight = 0.0;
    VecX m_deploymentWeight;
    VecX m_targetDeploymentHeight;

    std::vector<std::vector<size_t>> m_umbrella_to_top_bottom_joint_map;
    DeploymentForceType m_dftype = DeploymentForceType::LinearActuator;
    AngleBoundEnforcement m_angleBoundEnforcement = AngleBoundEnforcement::Hard;
    VecX m_deploymentEnergyWeight;
    LinearActuators<UmbrellaMesh_T> m_linearActuator;
    ConstraintBarrier m_constraintBarrier;

    // Repulsion
    std::vector<std::vector<size_t>> m_umbrella_connectivity;
    Real_ m_repulsionEnergyWeight = 0.0;

    struct SensitivityCache {
        SensitivityCache();

        // Cache of constrained terminal edges' Jacobians and Hessians
        // (to accelerate repeated calls to elastic energy Hessian/gradient).
        // The entries for segment si's two ends are at
        // sensitivityForTerminalEdge[2 * si + 0] and sensitivityForTerminalEdge[2 * si + 1]
        // (entries for free ends are left uninitialized).
        std::vector<UmbrellaMeshTerminalEdgeSensitivity<Real_>> sensitivityForTerminalEdge;

        bool evaluatedWithUpdatedSource = true;
        bool evaluatedHessian = false;
        void update(const UmbrellaMesh_T &l, bool updatedSource, bool evalHessian);
        // Compute directional derivative of Jacobian ("delta_jacobian") instead of the full Hessian
        void update(const UmbrellaMesh_T &l, bool updatedSource, const VecX &delta_params);

        const UmbrellaMeshTerminalEdgeSensitivity<Real_> &lookup(size_t si, TerminalEdge which) const;

        bool filled() const { return !sensitivityForTerminalEdge.empty(); }

        void clear();
        ~SensitivityCache();
    };
    mutable SensitivityCache m_sensitivityCache;

    ////////////////////////////////////////////////////////////////////////////
    // Cache for hessian sparsity patterns
    ////////////////////////////////////////////////////////////////////////////
    mutable std::unique_ptr<CSCMat> m_cachedHessianSparsity, m_cachedHessianVarRLSparsity, m_cachedHessianPARLSparsity;
    void m_clearCache() { m_cachedHessianSparsity.reset(), m_cachedHessianVarRLSparsity.reset(), m_cachedHessianPARLSparsity.reset(); }
};

#include "UmbrellaMeshSegmentGradientAssembly.inl"

#endif	/* end of include guard: UmbrellaMesh_HH */

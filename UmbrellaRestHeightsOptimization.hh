////////////////////////////////////////////////////////////////////////////////
// UmbrellaRestHeightsOptimization.hh
////////////////////////////////////////////////////////////////////////////////
#ifndef UMBRELLARESTHEIGHTSOPTIMIZATION_HH
#define UMBRELLARESTHEIGHTSOPTIMIZATION_HH

#include "UmbrellaMesh.hh"
#include "UmbrellaOptimization.hh"

struct UmbrellaRestHeightsOptimization {
    UmbrellaRestHeightsOptimization(UmbrellaOptimization &um_opt)
    : m_um_opt(um_opt) { 
        objective = um_opt.objective;
        m_constructUmbrellaRestHeightsToArmRestLenMapTranspose(); }

    // All the top rest length variables followed by all the bottom rest length variables.
    // TODO: add check for inconsistent arm rest lengths that correspond to the same umbrella height variable. Currently the last one wins.
    const Eigen::VectorXd params();

    // m_UmbrellaRestHeightsToPerArmRestLength
    Eigen::VectorXd applyTransformation(const Eigen::VectorXd &URH) const {
        return m_umbrellaRestHeightsToArmRestLenMapTranspose.apply(URH, /* transpose */ true);
    }
    // m_PerArmRestLengthToUmbrellaRestHeights
    Eigen::VectorXd applyTransformationTranspose(const Eigen::VectorXd &PARL) const {
        return m_umbrellaRestHeightsToArmRestLenMapTranspose.apply(PARL, /* transpose */ false);
    }

    // Evaluate at a new set of parameters and commit this change to the umbrella mesh (which
    // are used as a starting point for solving the line search equilibrium)
    void newPt(const Eigen::VectorXd &params) {
        get_parent_opt().newPt(applyTransformation(params));
    }

    // No linear inequality constraints (beyond the length bound constraints).
    size_t numLinearInequalityConstraints() const { return 0; }
    std::vector<LinearInequality> getLinearInequalityConstraints(Real /* minRestLen */) const { return std::vector<LinearInequality>(); }

    // Objective function definition.
    Real J() { return J(params()); }
    Real J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) { return get_parent_opt().J(applyTransformation(params), opt_eType); }
    // Gradient of the objective over the design parameters.
    Eigen::VectorXd gradp_J() { return gradp_J(params()); }
    Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) { return applyTransformationTranspose(get_parent_opt().gradp_J(applyTransformation(params), opt_eType)); }

    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, OptEnergyType opt_eType = OptEnergyType::Full) { return applyTransformationTranspose(get_parent_opt().apply_hess(applyTransformation(params), applyTransformation(delta_p), coeff_J, opt_eType)); }

    Eigen::VectorXd apply_hess_J               (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, OptEnergyType opt_eType = OptEnergyType::Full) { return apply_hess(params, delta_p, 1.0, opt_eType); }
    
    // We define two rest lengths variables per umbrella: one for the top three arms and one for the bottom three arms. 
    size_t numRestLen() const { return get_parent_opt().linesearchObject().numURH(); }
    // Currently we only have the rest lengths as design parameters.
    size_t numParams() const { return numRestLen(); }
    Real defaultLengthBound() const { return get_parent_opt().defaultLengthBound(); }

    const UmbrellaMesh &linesearchObject() const { return get_parent_opt().linesearchObject(); }
    const UmbrellaMesh &committedObject()  const { return get_parent_opt().committedObject(); }
    void setGamma(Real val)                             { get_parent_opt().setGamma(val); }
    Real getGamma()                        const { return get_parent_opt().getGamma(); }
    void setBeta(Real val)                              { get_parent_opt().setBeta (val); }
    Real getBeta()                         const { return get_parent_opt().getBeta (); }
    void setEta(Real val)                               { get_parent_opt().setEta  (val); }
    Real getEta()                          const { return get_parent_opt().getEta  (); }
    void setZeta(Real val)                              { get_parent_opt().setZeta (val); }
    Real getZeta()                         const { return get_parent_opt().getZeta (); }
    void setIota(Real val)                              { get_parent_opt().setIota (val); }
    Real getIota()                         const { return get_parent_opt().getIota (); }
    void invalidateAdjointState() { get_parent_opt().invalidateAdjointState(); }
    void reset_joint_target_with_closest_points() { get_parent_opt().reset_joint_target_with_closest_points(); }

    using DOO  = DesignOptimizationObjective<UmbrellaMesh_T, OptEnergyType>;
    DOO objective;

    UmbrellaOptimization &get_parent_opt() const { return m_um_opt; }
    ~UmbrellaRestHeightsOptimization() = default;

protected:
    void m_checkArmHeightValidity(const UmbrellaMesh &um) const;

    // Construct the *transpose* of the map from a vector holding the rest heights
    // of each umbrella to a vector holding a (rest) length for each arm segment in the
    // entire network. Assume in each umbrella, all top arms have the same length and all bottom arms have the same length.
    void m_constructUmbrellaRestHeightsToArmRestLenMapTranspose();

    UmbrellaOptimization &m_um_opt;
    SuiteSparseMatrix m_umbrellaRestHeightsToArmRestLenMapTranspose; // Non-autodiff! (The map is piecewise constant/nondifferentiable).

};

#endif /* end of include guard: UMBRELLARESTHEIGHTSOPTIMIZATION_HH */

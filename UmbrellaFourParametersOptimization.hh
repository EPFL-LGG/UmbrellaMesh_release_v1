////////////////////////////////////////////////////////////////////////////////
// UmbrellaFourParametersOptimization.hh
////////////////////////////////////////////////////////////////////////////////
#ifndef UMBRELLAFOURPARAMETERSOPTIMIZATION_HH
#define UMBRELLAFOURPARAMETERSOPTIMIZATION_HH

#include "UmbrellaMesh.hh"
#include "UmbrellaOptimization.hh"

struct UmbrellaFourParametersOptimization {
    UmbrellaFourParametersOptimization(UmbrellaOptimization &um_opt)
    : m_um_opt(um_opt) { 
        objective = um_opt.objective;
        m_constructUmbrellaFourParametersToArmRestLenMapTranspose(); 
        m_setEmptyArmParams();
    }

    // l1, l1, ..., l1, l2, ..., l2, l3, ..., l3, H, ..., H
    const Eigen::VectorXd params();

    // m_UmbrellaFourParametersToPerArmRestLength
    Eigen::VectorXd applyTransformation(const Eigen::VectorXd &URFP) const {
        return m_umbrellaFourParametersToArmRestLenMapTranspose.apply(URFP, /* transpose */ true);
    }
    // m_PerArmRestLengthToUmbrellaFourParameters
    Eigen::VectorXd applyTransformationTranspose(const Eigen::VectorXd &PARL) const {
        return m_umbrellaFourParametersToArmRestLenMapTranspose.apply(PARL, /* transpose */ false);
    }

    // Evaluate at a new set of parameters and commit this change to the umbrella mesh (which
    // are used as a starting point for solving the line search equilibrium)
    void newPt(const Eigen::VectorXd &params) {
        get_parent_opt().newPt(applyTransformation(params));
    }

    size_t numUmbrellas() const { return linesearchObject().numUmbrellas(); }
    size_t numLinearInequalityConstraints() const { return 3 * numUmbrellas() +  m_emptyArmParams.size(); }
    struct LinearInequality {
        std::vector<int>  vars;
        std::vector<Real> coeffs;
        Real              constPart;
    };
    // Get three constraints of the form "H - l - lmin >= 0" for each umbrella.
    std::vector<LinearInequality> getLinearInequalityConstraints(Real minRestLen) const {
        const int nu = numUmbrellas();
        std::vector<LinearInequality> result(3 * nu + m_emptyArmParams.size());
        for (int ui = 0; ui < nu; ++ui) {
            for (int l = 0; l < 3; ++l) {
                size_t ei = m_emptyArmParamIndex(ui + l * nu);
                if (ei < m_emptyArmParams.size()) {
                    result[3 * ui + l].vars      = { ui + l * nu };
                    result[3 * ui + l].coeffs    = {         1.0 };
                    result[3 * ui + l].constPart = 0;

                    result[3 * nu + ei].vars      = { ui + l * nu };
                    result[3 * nu + ei].coeffs    = {        -1.0 };
                    result[3 * nu + ei].constPart = 0;
                } else {
                    result[3 * ui + l].vars      = { ui + l * nu, ui + 3 * nu };
                    result[3 * ui + l].coeffs    = {        -1.0,         1.0 };
                    result[3 * ui + l].constPart = -minRestLen;
                }
            }
        }
        return result;
    }

    // Objective function definition.
    Real J() { return J(params()); }
    Real J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) { return get_parent_opt().J(applyTransformation(params), opt_eType); }
    // Gradient of the objective over the design parameters.
    Eigen::VectorXd gradp_J() { return gradp_J(params()); }
    Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) { return applyTransformationTranspose(get_parent_opt().gradp_J(applyTransformation(params), opt_eType)); }

    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, OptEnergyType opt_eType = OptEnergyType::Full) { return applyTransformationTranspose(get_parent_opt().apply_hess(applyTransformation(params), applyTransformation(delta_p), coeff_J, opt_eType)); }

    Eigen::VectorXd apply_hess_J               (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, OptEnergyType opt_eType = OptEnergyType::Full) { return apply_hess(params, delta_p, 1.0, opt_eType); }
    
    // We define three rest lengths variables for the top arms of the umbrella, but the height variables should have the same lower bound anyway.
    size_t numRestLen() const { return linesearchObject().numUmbrellas() * 4; }
    // The design parameters are the rest length of the top arms and the heights of the umbrellas.
    size_t numParams() const { return linesearchObject().numUmbrellas() * 4; }
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
    ~UmbrellaFourParametersOptimization() = default;

protected:
    void m_checkArmHeightValidity(const UmbrellaMesh &um) const;

    // Construct the *transpose* of the map from a vector holding the rest heights
    // of each umbrella to a vector holding a (rest) length for each arm segment in the
    // entire network. Assume in each umbrella, all top arms have the same length and all bottom arms have the same length.
    void m_constructUmbrellaFourParametersToArmRestLenMapTranspose();
    std::vector<size_t> m_setEmptyArmParams();
    int m_emptyArmParamIndex(size_t pi) const {
        return std::distance(m_emptyArmParams.begin(), std::find(m_emptyArmParams.begin(), m_emptyArmParams.end(), pi));
    }
    UmbrellaOptimization &m_um_opt;
    SuiteSparseMatrix m_umbrellaFourParametersToArmRestLenMapTranspose; // Non-autodiff! (The map is piecewise constant/nondifferentiable).
    std::vector<size_t> m_emptyArmParams;

};

#endif /* end of include guard: UMBRELLAFOURPARAMETERSOPTIMIZATION_HH */

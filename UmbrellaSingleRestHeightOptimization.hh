////////////////////////////////////////////////////////////////////////////////
// UmbrellaSingleRestHeightOptimization.hh
////////////////////////////////////////////////////////////////////////////////
#ifndef UMBRELLASINGLERESTHEIGHTOPTIMIZATION_HH
#define UMBRELLASINGLERESTHEIGHTOPTIMIZATION_HH

#include "UmbrellaMesh.hh"
#include "UmbrellaRestHeightsOptimization.hh"

struct UmbrellaSingleRestHeightOptimization {
    UmbrellaSingleRestHeightOptimization(UmbrellaRestHeightsOptimization &um_rh_opt)
    : m_um_rh_opt(um_rh_opt) { 
        objective = um_rh_opt.objective;
        m_constructUmbrellaSingleRestHeightToRestHeightsMapTranspose(); }

    const Eigen::VectorXd params() { return get_parent_opt().params().head(numParams()); }

    // m_SingleRestHeightToRestHeights
    Eigen::VectorXd applyTransformation(const Eigen::VectorXd &URH) const {
        return m_umbrellaSingleRestHeightToRestHeightsMapTranspose.apply(URH, /* transpose */ true);
    }
    // m_RestHeightsToSingleRestHeight
    Eigen::VectorXd applyTransformationTranspose(const Eigen::VectorXd &USRH) const {
        return m_umbrellaSingleRestHeightToRestHeightsMapTranspose.apply(USRH, /* transpose */ false);
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
    
    // Currently we only have a rest length per umbrella as design parameters.
    size_t numRestLen() const { return get_parent_opt().linesearchObject().numUmbrellas(); }
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

    UmbrellaRestHeightsOptimization &get_parent_opt() const { return m_um_rh_opt; }
    ~UmbrellaSingleRestHeightOptimization() = default;

protected:
    // Construct the *transpose* of the map from a vector holding the rest heights
    // of each umbrella to a vector holding a (rest) length for each arm segment in the
    // entire network. Assume in each umbrella, all top arms have the same length and all bottom arms have the same length.
    void m_constructUmbrellaSingleRestHeightToRestHeightsMapTranspose() {
        auto & um = m_um_rh_opt.committedObject();
        const SuiteSparse_long m = um.numUmbrellas(), n = um.numURH();
        SuiteSparseMatrix result(m, n);
        result.nz = um.numURH();

        // Now we fill out the transpose of the map one column (arm segment) at a time:
        //    #     [               ]
        // umbrella [               ]
        //           # umbrella * 2

        result.Ax.assign(result.nz, 1);
        auto &Ai = result.Ai;
        auto &Ap = result.Ap;

        Ai.reserve(result.nz);
        Ap.reserve(n + 1);

        Ap.push_back(0); // col 0 begin

        for (size_t ai = 0; ai < um.numURH(); ++ai) {
            Ai.push_back(ai % m);
            Ap.push_back(Ai.size()); // col end
        }

        assert(Ai.size() == size_t(result.nz));
        assert(Ap.size() == size_t(n + 1    ));

        m_umbrellaSingleRestHeightToRestHeightsMapTranspose = std::move(result);
    }

    UmbrellaRestHeightsOptimization &m_um_rh_opt;
    SuiteSparseMatrix m_umbrellaSingleRestHeightToRestHeightsMapTranspose; // Non-autodiff! (The map is piecewise constant/nondifferentiable).

};

#endif /* end of include guard: UMBRELLASINGLERESTHEIGHTOPTIMIZATION_HH */

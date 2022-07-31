////////////////////////////////////////////////////////////////////////////////
// UmbrellaOptimization.hh
////////////////////////////////////////////////////////////////////////////////
#ifndef UMBRELLAOPTIMIZATION_HH
#define UMBRELLAOPTIMIZATION_HH

#include "UmbrellaMesh.hh"
#include <MeshFEM/Geometry.hh>
#include "umbrella_compute_equilibrium.hh"
#include "UmbrellaDesignOptimizationTerms.hh"
#include "UmbrellaTargetSurfaceFitter.hh"

enum class OptAlgorithm    : int { NEWTON_CG=0, BFGS=1 };
enum class OptEnergyType   : int { Full, Elastic, Target, DeploymentForce, Stress, UmbrellaForces };
enum class PredictionOrder : int { Zero = 0, One = 1, Two = 2};

struct LinearInequality {
    std::vector<int>  vars;
    std::vector<Real> coeffs;
    Real              constPart;
};

struct UmbrellaOptimization {
    UmbrellaOptimization(UmbrellaMesh &umbrella, const NewtonOptimizerOptions &eopts, const Real elasticEnergyIncreaseFactorLimit, int pinJoint, bool useFixedJoint, const std::vector<size_t> &fixedVars);

    // Evaluate at a new set of parameters and commit this change to the umbrella mesh (which
    // are used as a starting point for solving the line search equilibrium)
    void newPt(const Eigen::VectorXd &params) {
        std::cout << "newPt at dist " << (m_umbrella.getDesignParameters() - params).norm() << std::endl;
        m_updateAdjointState(params); // update the adjoint state as well as the equilibrium, since we'll probably be needing gradients at this point.
        commitLinesearchUmbrella();
    }

    size_t numParams() const { return m_numParams; }
    size_t numRestLen() const { return m_linesearch_umbrella.numArmSegments(); }
    const Eigen::VectorXd &params() const { return m_umbrella.getDesignParameters(); }

    // No linear inequality constraints (beyond the length bound constraints).
    size_t numLinearInequalityConstraints() const { return 0; }
    std::vector<LinearInequality> getLinearInequalityConstraints(Real /* minRestLen */) const { return std::vector<LinearInequality>(); }

    // Objective function definition.
    Real J()        { return J(params()); }
    Real J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full);
    // Target fitting objective definition.
    Real J_target() { return J_target(params()); }
    Real J_target(const Eigen::Ref<const Eigen::VectorXd> &params);
    // Gradient of the objective over the design parameters.
    Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full);
    Eigen::VectorXd gradp_J()        { return gradp_J(params()); }

    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess_J(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, OptEnergyType opt_eType = OptEnergyType::Full) { return apply_hess(params, delta_p, 1.0, opt_eType); }
    Eigen::VectorXd apply_hess  (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, OptEnergyType opt_eType = OptEnergyType::Full);

    // Access members and quantities.
    Real defaultLengthBound() const { return 0.125 * m_umbrella.getPerArmRestLength().minCoeff(); }
    Real get_rl0() const { return m_rl0; }
    Real get_E0() const { return m_U0; }
    bool use_restKappa() { return m_linesearch_umbrella.getDesignParameterConfig().restKappa; }
    bool use_restLen()   { return m_linesearch_umbrella.getDesignParameterConfig().restLen; }
    // Get the index of the joint whose orientation is constrained to pin
    // down the linkage's rigid motion.
    size_t getRigidMotionConstrainedJoint() const { return m_rm_constrained_joint; }
    NewtonOptimizer &getEquilibriumOptimizer() { return *m_equilibrium_optimizer; }

    // Equilibrium Helpers.
    void commitLinesearchUmbrella() {
        m_umbrella.set(m_linesearch_umbrella);
        // Stash the current factorizations to be reused at each step of the linesearch
        // to predict the equilibrium at the new design parameters.
        getEquilibriumOptimizer().solver.stashFactorization();
        m_committed_ws = m_linesearch_ws->clone();
    }
    void setEquilibriumOptions(const NewtonOptimizerOptions &eopts);
    NewtonOptimizerOptions getEquilibriumOptions() const;

    // When the fitting weights change the adjoint state must be recomputed.
    // Let the user manually inform us of this change.
    void invalidateAdjointState() { m_adjointStateIsCurrent.first = false; }

    // Configure whether the closest point projections are held fixed in the
    // target attraction term of the simulation energy (`attractionHCP`) and in
    // the target-fitting term of the design objective (`objectiveHCP`).
    void setHoldClosestPointsFixed(bool attractionHCP, bool objectiveHCP) {
        if (attractionHCP != m_umbrella.getHoldClosestPointsFixed()) {
            std::cout << "Updating m_umbrella hcp" << std::endl;
            m_umbrella.setHoldClosestPointsFixed(attractionHCP);
            invalidateEquilibria();
        }

        if (objectiveHCP != target_surface_fitter->holdClosestPointsFixed) {
            std::cout << "Updating tsf hcp" << std::endl;
            target_surface_fitter->holdClosestPointsFixed = objectiveHCP;
            invalidateAdjointState();
        }

        std::cout << "hold closest points fixed: " << m_umbrella.getHoldClosestPointsFixed() << ", " << target_surface_fitter->holdClosestPointsFixed << std::endl;
    }

    void reset_joint_target_with_closest_points() {
        m_umbrella.reset_joint_target_with_closest_points();
        target_surface_fitter->reset_joint_target_with_closest_points(m_umbrella);
        invalidateEquilibria();
    }
    // To be called after modifying the simulation energy formulation.
    // In this case, the committed and linesearch equilibra, as well as any
    // cached Hessian factorizations and adjoint states, are invalid and must
    // be recomputed.
    void invalidateEquilibria() {
        std::cout << "Invalidating equilibria" << std::endl;
        const Eigen::VectorXd  committedParams =            m_umbrella.getDesignParameters();
        const Eigen::VectorXd linesearchParams = m_linesearch_umbrella.getDesignParameters();

        getEquilibriumOptimizer().solver.clearStashedFactorization();
        m_linesearch_umbrella.set(m_umbrella);
        m_forceEquilibriumUpdate();           // recompute the equilibrium for the commited design parameters; done without any prediction
        commitLinesearchUmbrella();           // commit the updated equilibrium, stashing the updated factorization
        m_updateEquilibria(linesearchParams); // recompute the linesearch equilibrium (if the linesearch params differ from the committed ones).
    }

    void setAttractionWeight(Real attraction_weight) {
        m_umbrella.setAttractionWeight(attraction_weight);
        invalidateEquilibria();
    }
    Real getAttractionWeight() { return m_umbrella.getAttractionWeight(); }

    const UmbrellaMesh &    linesearchObject() const { return m_linesearch_umbrella; }
    const UmbrellaMesh &     committedObject() const { return m_umbrella; }
    const WorkingSet   &linesearchWorkingSet() const { if (!m_linesearch_ws) throw std::runtime_error("Unallocated working set"); return *m_linesearch_ws; }
    const WorkingSet   & committedWorkingSet() const { if ( !m_committed_ws) throw std::runtime_error("Unallocated working set"); return  *m_committed_ws; }

    void setGamma(Real val) { objective.get("ElasticEnergy").setWeight(val / m_U0);                      invalidateAdjointState();}
    Real getGamma() const   { return objective.get("ElasticEnergy").getWeight() * m_U0; }
    void setBeta(Real val)  { beta = val; objective.get("TargetFitting").setWeight(val / (m_l0 * m_l0)); invalidateAdjointState(); }
    Real getBeta() const    { return objective.get("TargetFitting").getWeight() * m_l0 * m_l0; }
    void setEta(Real val)   { eta = val; objective.get("DeploymentForce").setWeight(val / (m_l0 * m_l0)); invalidateAdjointState(); }
    Real getEta() const     { return objective.get("DeploymentForce").getWeight() * m_l0 * m_l0; }
    void setZeta(Real val)  { zeta = val; objective.get("Stress").setWeight(val / (m_l0 * m_l0)); invalidateAdjointState(); }
    Real getZeta() const    { return objective.get("Stress").getWeight() * m_l0 * m_l0; }
    void setIota(Real val)  { iota = val; objective.get("UmbrellaForces").setWeight(val / (m_l0 * m_l0)); invalidateAdjointState(); }
    Real getIota() const    { return objective.get("UmbrellaForces").getWeight() * m_l0 * m_l0; }
    using DOO  = DesignOptimizationObjective<UmbrellaMesh_T, OptEnergyType>;
    using TRec = typename DOO::TermRecord;
    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    DOO objective;
    std::shared_ptr<TargetSurfaceFitter> target_surface_fitter;
    // Configure how the equilibrium at a perturbed set of parameters is predicted (using 0th, 1st, or 2nd order Taylor expansion)
    PredictionOrder prediction_order = PredictionOrder::One;

    ~UmbrellaOptimization() = default;

    // Debugging
    // Access adjoint state for debugging
    Eigen::VectorXd get_w() const { return objective.adjointState(); }
    Eigen::VectorXd get_w_rhs() const { return m_w_rhs; }
    Eigen::VectorXd get_delta_w_rhs() const { return m_delta_w_rhs; }

    Eigen::VectorXd get_delta_x() const { return m_delta_x; }
    Eigen::VectorXd get_delta_w() const { return m_delta_w; }
    Eigen::VectorXd get_d3E_w() const { return m_d3E_w; }

    Eigen::VectorXd get_delta_delta_x () const { return m_delta_delta_x; }
    Eigen::VectorXd get_second_order_x() const { return m_second_order_x; }
    Eigen::VectorXd get_H_times(Eigen::VectorXd &w) const { return m_linesearch_umbrella.applyHessian(w); }

protected:
    void m_forceEquilibriumUpdate();
    // Return whether "params" are actually new...
    bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &params);

    // Update the adjoint state vectors "w" and "y"
    bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full);

    ////////////////////////////////////////////////////////////////////////////
    // Private member variables
    ////////////////////////////////////////////////////////////////////////////
    NewtonOptimizerOptions m_equilibrium_options;
    std::vector<size_t> m_fixedVars;
    size_t m_numParams;
    size_t m_rm_constrained_joint; // index of joint whose rigid motion is constrained.
    Real m_U0 = 1.0, m_l0 = 1.0, m_rl0 = 1.0;
    Real beta = 1.0;
    Real eta = 1.0;
    Real zeta = 1.0;
    Real iota = 1.0;
    Real m_alpha_tgt = 0.0;

    UmbrellaMesh &m_umbrella;
    UmbrellaMesh m_linesearch_umbrella;

    bool m_autodiffUmbrellaIsCurrent = false;
    std::pair<bool, OptEnergyType> m_adjointStateIsCurrent;
    bool m_equilibriumSolveSuccessful = false;

    std::unique_ptr<NewtonOptimizer> m_equilibrium_optimizer;
    std::unique_ptr<WorkingSet> m_linesearch_ws, m_committed_ws; // Working set at the linesearch/committed linkage

    UmbrellaMesh_T<ADReal> m_diff_umbrella;

    Eigen::VectorXd m_delta_w, m_delta_x; // variations of adjoint/forward state from the last call to apply_hess (for debugging)
    Eigen::VectorXd m_delta_delta_x;   // second variations of forward state from last call to m_updateEquilibrium (for debugging)
    Eigen::VectorXd m_second_order_x; // second-order predictions of the linkage's equilibrium (for debugging)
    Eigen::VectorXd m_d3E_w;          // analytical derivative of Hw with w held fixed
    Eigen::VectorXd m_w_rhs, m_delta_w_rhs;
};

#endif /* end of include guard: UMBRELLAOPTIMIZATION_HH */

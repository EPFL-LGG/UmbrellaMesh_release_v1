#include "UmbrellaOptimization.hh"

UmbrellaOptimization::UmbrellaOptimization(UmbrellaMesh &umbrella, const NewtonOptimizerOptions &eopts, const Real elasticEnergyIncreaseFactorLimit, int pinJoint, bool useFixedJoint, const std::vector<size_t> &fixedVars)
    : m_equilibrium_options(eopts), m_numParams(umbrella.numDesignParams()),  m_U0(umbrella.energyElastic()), m_l0(BBox<Point3D>(umbrella.deformedPoints()).dimensions().norm()), m_umbrella(umbrella), m_linesearch_umbrella(umbrella) {
        std::runtime_error mismatch("Linkage mismatch");
    if (m_numParams != umbrella.numDesignParams()) throw mismatch;

    // Initialize the auto diff
    m_diff_umbrella.set(umbrella);

    bool use_targetFitting = m_umbrella.hasTargetSurface();
    if (use_targetFitting) {
        target_surface_fitter = m_umbrella.getTargetSurface()->clone();
        target_surface_fitter->forceUpdateClosestPoints(m_umbrella);
    }
    else {
        std::cout<<"Warning: The input umbrella mesh has no target surface fitter object!"<<std::endl;
    }

    // Create the objective terms
    using OET = OptEnergyType;
    using EEO = ElasticEnergyObjective<UmbrellaMesh_T>;
    using TFO = TargetFittingDOOT<UmbrellaMesh_T>;
    using DFO = DeploymentForceDOOT<UmbrellaMesh_T>;
    using LSO = LpStressDOOT<UmbrellaMesh_T>;
    using UFO = UmbrellaForceObjective<UmbrellaMesh_T>;
    objective.add("ElasticEnergy", OET::Elastic, std::make_shared<EEO>(m_linesearch_umbrella), 1.0 / m_U0);

    if (use_targetFitting)
        objective.add("TargetFitting", OET::Target, std::make_shared<TFO>(m_linesearch_umbrella, *target_surface_fitter), beta / (m_l0 * m_l0));

    objective.add("DeploymentForce", OET::DeploymentForce, std::make_shared<DFO>(m_linesearch_umbrella), eta / (m_l0 * m_l0));
    objective.add("Stress",          OET::Stress         , std::make_unique<LSO>(m_linesearch_umbrella), zeta);
    objective.add("UmbrellaForces",  OET::UmbrellaForces , std::make_unique<UFO>(m_linesearch_umbrella), iota);

    m_fixedVars.insert(std::end(m_fixedVars), std::begin(fixedVars), std::end(fixedVars));

    if (useFixedJoint) {
        // Constrain the position and orientation of the centermost joint to prevent global rigid motion.
        if (pinJoint != -1) {
            m_rm_constrained_joint = pinJoint;
            if (m_rm_constrained_joint >= umbrella.numJoints()) throw std::runtime_error("Manually specified pinJoint is out of bounds");
        }
        else {
            m_rm_constrained_joint = umbrella.centralJoint();
        }
        const size_t jdo = umbrella.dofOffsetForJoint(m_rm_constrained_joint);
        for (size_t i = 0; i < 6; ++i) m_fixedVars.push_back(jdo + i);
    }
    m_equilibrium_optimizer = get_equilibrium_optimizer(m_linesearch_umbrella, TARGET_ANGLE_NONE, m_fixedVars);
    m_equilibrium_optimizer->options = m_equilibrium_options;
    dynamic_cast<EquilibriumProblem<UmbrellaMesh> &>(m_equilibrium_optimizer->get_problem()).elasticEnergyIncreaseFactorLimit = elasticEnergyIncreaseFactorLimit;

    // Ensure we start at an equilibrium (using the passed equilibrium solver options)
    m_forceEquilibriumUpdate();
    commitLinesearchUmbrella();
}


////////////////////////////////////////////////////////////////////////////
// Equilibrium
////////////////////////////////////////////////////////////////////////////

void UmbrellaOptimization::m_forceEquilibriumUpdate() {
    m_equilibriumSolveSuccessful = true;
    // Initialize the working set by copying the committed linkage's working
    // set if it exists; otherwise, construct a fresh one.
    if (m_committed_ws) m_linesearch_ws = m_committed_ws->clone();
    else                m_linesearch_ws = std::make_unique<WorkingSet>(getEquilibriumOptimizer().get_problem());
    try {
        if (m_equilibrium_options.verbose)
            std::cout << "Umbrella Mesh equilibrium solve" << std::endl;
        auto cr = getEquilibriumOptimizer().optimize(*m_linesearch_ws);
        // A backtracking failure will happen if the gradient tolerance is set too low
        // and generally does not indicate a complete failure/bad estimate of the equilibrium.
        // We therefore accept such equilibria with a warning.
        bool acceptable_failed_equilibrium = cr.backtracking_failure;
        // std::cout << "cr.backtracking_failure: " << cr.backtracking_failure << std::endl;
        // std::cout << "cr.indefinite.back(): " << cr.indefinite.back() << std::endl;
        if (!cr.success && !acceptable_failed_equilibrium) {
            throw std::runtime_error("Equilibrium solve did not converge");
        }
        if (acceptable_failed_equilibrium) {
            std::cout << "WARNING: equillibrium solve backtracking failure." << std::endl;
        }
    }
    catch (const std::runtime_error &e) {
        std::cout << "Equilibrium solve failed: " << e.what() << std::endl;
        m_equilibriumSolveSuccessful = false;
        return; // subsequent update_factorizations will fail if we caught a Tau runaway...
    }

    auto &uo = getEquilibriumOptimizer();
    // Factorize the Hessian for the updated linesearch equilibrium (not the second-to-last iteration).
    if (uo.solver.hasStashedFactorization() &&
            (m_linesearch_umbrella.getDesignParameters() - m_umbrella.getDesignParameters()).squaredNorm() == 0) {
        // We are re-evaluating at the committed equilibrium; roll back to the committed factorization
        // rather than re-computing it!
        uo.solver.swapStashedFactorization(); // restore the committed factorization
        uo.solver.      stashFactorization(); // stash a copy of it (TODO: avoid copy?)
    }
    else {
        // Use the final equilibrium's Hessian for sensitivity analysis, not the second-to-last iterates'
        try {
            getEquilibriumOptimizer().update_factorizations(*m_linesearch_ws);
        }
        catch (const std::runtime_error &e) {
            std::cout << "Hessian factorization at equilibrium failed failed: " << e.what() << std::endl;
            m_equilibriumSolveSuccessful = false;
            return;
        }
    }

    // The cached adjoint state is invalidated whenever the equilibrium is updated...
    m_adjointStateIsCurrent.first = false;
    m_autodiffUmbrellaIsCurrent   = false;

    objective.update();
}


bool UmbrellaOptimization::m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &newParams) {
    if (size_t(newParams.size()) != numParams()) throw std::runtime_error("Parameter vector size mismatch");
    if ((m_linesearch_umbrella.getDesignParameters() - newParams).norm() < 1e-16) return false;
    m_linesearch_umbrella.set(m_umbrella);

    const Eigen::VectorXd currParams = m_umbrella.getDesignParameters();
    Eigen::VectorXd delta_p = newParams - currParams;

    if (delta_p.squaredNorm() == 0) { // returning to linesearch start; no prediction/Hessian factorization necessary
        m_forceEquilibriumUpdate();
        return true;
    }

    // Apply the new design parameters and measure the energy with the 0^th order prediction
    // (i.e. at the currently committed equilibrium).
    // We will only replace this equilibrium if the higher-order predictions achieve a lower energy.
    m_linesearch_umbrella.setDesignParameters(newParams);
    Real bestEnergy = m_linesearch_umbrella.energy();
    Eigen::VectorXd curr_x = m_umbrella.getDoFs();
    Eigen::VectorXd best_x = curr_x;
    if (prediction_order > PredictionOrder::Zero) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Predict equilibrium");
        // Return to using the Hessian for the last committed linkage
        // (i.e. for the equilibrium stored in m_flat and m_deployed).
        auto &opt_umbrella = getEquilibriumOptimizer();
        if (!opt_umbrella.solver.hasStashedFactorization())
            throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");
        opt_umbrella.solver.swapStashedFactorization();

        {
            // Solve for equilibrium perturbation corresponding to delta_p:
            //      [H][delta x] = [-d2E/dxdp delta_p]
            //                     \_________________/
            //                              b
            const size_t np = numParams(), nd = m_umbrella.numDoF();
            VecX_T<Real> neg_deltap_padded(nd + np);
            neg_deltap_padded.setZero();
            neg_deltap_padded.tail(np) = -delta_p;

            // Computing -d2E/dxdp delta_p can skip the *-x and designParameter-* blocks
            HessianComputationMask mask_dxdp;
            mask_dxdp.dof_in              = false;
            mask_dxdp.designParameter_out = false;

            auto neg_d2E_dxdp_deltap = m_umbrella.applyHessianPerArmRestlen(neg_deltap_padded, mask_dxdp).head(nd);
            m_committed_ws->getFreeComponentInPlace(neg_d2E_dxdp_deltap); // Enforce any active bound constraints (equilibrium optimizer already modified the Hessian accordingly)
            m_delta_x = opt_umbrella.extractFullSolution(opt_umbrella.solver.solve(opt_umbrella.removeFixedEntries(neg_d2E_dxdp_deltap)));
            m_committed_ws->validateStep(m_delta_x); // Debugging: ensure bound constraints are respected perfectly.

            // Evaluate the energy at the 1st order-predicted equilibrium
            {
                auto first_order_x = (curr_x + m_delta_x).eval();
                m_linesearch_umbrella.setDoFs(first_order_x);
                Real energy1stOrder = m_linesearch_umbrella.energy();
                if (energy1stOrder < bestEnergy) { std::cout << " used first order prediction, energy reduction " << bestEnergy - energy1stOrder << std::endl; bestEnergy = energy1stOrder; best_x = first_order_x; } else { m_linesearch_umbrella.setDoFs(best_x); }
            }

            if (prediction_order > PredictionOrder::One) {
                // Solve for perturbation of equilibrium perturbation corresponding to delta_p:
                //      H (delta_p^T d2x/dp^2 delta_p) = -(d3E/dx3 delta_x + d3E/dx2dp delta_p) delta_x -(d3E/dxdpdx delta_x + d3E/dxdpdp delta_p) delta_p
                //                                     = -directional_derivative_delta_dxp(d2E/dx2 delta_x + d2E/dxdp delta_p)  (Directional derivative computed with autodiff)
                m_diff_umbrella.set(m_umbrella);

                Eigen::VectorXd neg_d3E_delta_x;
                {
                    VecX_T<Real> delta_xp(nd + np);
                    delta_xp << m_delta_x, delta_p;

                    // inject equilibrium and design parameter perturbation
                    VecX_T<ADReal> ad_xp = m_umbrella.getExtendedDoFsPARL();
                    for (size_t i = 0; i < nd + np; ++i) ad_xp[i].derivatives()[0] = delta_xp[i];
                    m_diff_umbrella.setExtendedDoFsPARL(ad_xp);

                    neg_d3E_delta_x = -extractDirectionalDerivative(m_diff_umbrella.applyHessianPerArmRestlen(delta_xp)).head(nd);
                }

                m_committed_ws->getFreeComponentInPlace(neg_d3E_delta_x); // Enforce any active bound constraints.
                m_delta_delta_x = opt_umbrella.extractFullSolution(opt_umbrella.solver.solve(opt_umbrella.removeFixedEntries(neg_d3E_delta_x)));
                m_committed_ws->validateStep(m_delta_delta_x); // Debugging: ensure bound constraints are respected perfectly.

                // Evaluate the energy at the 2nd order-predicted equilibrium, roll back to previous best if energy is higher.
                {
                    m_second_order_x = (curr_x + m_delta_x + 0.5 * m_delta_delta_x).eval(),
                    m_linesearch_umbrella.setDoFs(m_second_order_x);
                    Real energy2ndOrder = m_linesearch_umbrella.energy();
                    if (energy2ndOrder < bestEnergy) { std::cout << " used second order prediction, energy reduction " << bestEnergy - energy2ndOrder << std::endl; bestEnergy = energy2ndOrder; best_x = m_second_order_x;} else { m_linesearch_umbrella.setDoFs(best_x); }
                }
            }
        }

        // Return to using the primary factorization, storing the committed
        // linkages' factorizations back in the stash for later use.
        opt_umbrella.solver.swapStashedFactorization();
    }
    m_forceEquilibriumUpdate();
    return true;
}

////////////////////////////////////////////////////////////////////////////
// Energy, Gradient, Hessian Vector Product
////////////////////////////////////////////////////////////////////////////

Real UmbrellaOptimization::J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType) {
    std::cout << "eval at dist " <<            (m_umbrella.getDesignParameters() - params).norm() << std::endl;
    std::cout << "eval at linesearch dist " << (m_linesearch_umbrella.getDesignParameters() - params).norm() << std::endl;
    m_updateEquilibria(params);
    if (!m_equilibriumSolveSuccessful) return std::numeric_limits<Real>::max();

    // objective.printReport(); // Note: this forces the evaluation of 0-weight terms (which may be expensive)
    return objective.value(opt_eType);
}

Real UmbrellaOptimization::J_target(const Eigen::Ref<const Eigen::VectorXd> &params) {
    m_updateEquilibria(params);
    return objective.value("TargetFitting");
}

bool UmbrellaOptimization::m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType) {
    m_updateEquilibria(params);
    if (m_adjointStateIsCurrent.first && (m_adjointStateIsCurrent.second == opt_eType)) return false;
    std::cout << "Updating adjoint state" << std::endl;

    // Solve the adjoint problems needed to efficiently evaluate the gradient.
    // Note: if the Hessian modification failed (tau runaway), the adjoint state
    // solves will fail. To keep the solver from giving up entirely, we simply
    // set the adjoint state to 0 in these cases. Presumably this only happens
    // at bad iterates that will be discarded anyway.
    try {
        // Adjoint solve for the target fitting objective on the deployed linkage
        objective.updateAdjointState(getEquilibriumOptimizer(), *m_linesearch_ws, opt_eType);
    }
    catch (...) {
        std::cout << "WARNING: Adjoint state solve failed" << std::endl;
        objective.clearAdjointState();
    }

    m_adjointStateIsCurrent = std::make_pair(true, opt_eType);

    return true;
}

Eigen::VectorXd UmbrellaOptimization::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType) {
    m_updateAdjointState(params, opt_eType);
    m_w_rhs = objective.grad_x(opt_eType); // Debugging

    HessianComputationMask mask;
    mask.dof_out = false;
    mask.designParameter_in = false;

    const size_t nd = m_linesearch_umbrella.numDoF();
    const size_t np = numParams();
    Eigen::VectorXd w_padded(nd + np);
    w_padded.head(nd) = objective.adjointState();
    w_padded.tail(np).setZero();
    return objective.grad_p(opt_eType) - m_linesearch_umbrella.applyHessianPerArmRestlen(w_padded, mask).tail(numParams());
}


Eigen::VectorXd UmbrellaOptimization::apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params,
                                                 const Eigen::Ref<const Eigen::VectorXd> &delta_p,
                                                 Real coeff_J, OptEnergyType opt_eType) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("apply_hess_J");
    BENCHMARK_START_TIMER_SECTION("Preamble");
    const size_t np = numParams(), nd = m_linesearch_umbrella.numDoF();
    if (size_t( params.size()) != np) throw std::runtime_error("Incorrect parameter vector size");
    if (size_t(delta_p.size()) != np) throw std::runtime_error("Incorrect delta parameter vector size");
    m_updateAdjointState(params, opt_eType);

    if (!m_autodiffUmbrellaIsCurrent) {
        BENCHMARK_SCOPED_TIMER_SECTION timer2("Update autodiff linkages");
        m_diff_umbrella.set(m_linesearch_umbrella);
        m_autodiffUmbrellaIsCurrent = true;
    }

    auto &opt = getEquilibriumOptimizer();
    auto &H   = opt.solver;

    BENCHMARK_STOP_TIMER_SECTION("Preamble");

    VecX_T<Real> neg_deltap_padded(nd + np);
    neg_deltap_padded.head(nd).setZero();
    neg_deltap_padded.tail(np) = -delta_p;

    // Computing -d2E/dxdp delta_p can skip the *-x and designParameter-* blocks
    HessianComputationMask mask_dxdp, mask_dxpdx;
    mask_dxdp.dof_in              = false;
    mask_dxdp.designParameter_out = false;
    mask_dxpdx.designParameter_in = false;

    VecX_T<Real> delta_dJ_dxp;

    try {
        // Solve for state perturbation
        // H delta x = [-d2E/dxdp delta_p]
        //             \_________________/
        //                     b
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta x");
            VecX_T<Real> b = m_linesearch_umbrella.applyHessianPerArmRestlen(neg_deltap_padded, mask_dxdp).head(nd);
            m_linesearch_ws->getFreeComponentInPlace(b); // Enforce any active bound constraints (equilibrium optimizer already modified the Hessian accordingly)
            m_delta_x = opt.extractFullSolution(H.solve(opt.removeFixedEntries(b)));
            m_linesearch_ws->validateStep(m_delta_x); // Debugging: ensure bound constraints are respected perfectly.
        }

        VecX_T<Real> delta_xp(nd + np);
        delta_xp << m_delta_x, delta_p;

        // Solve for adjoint state perturbation
        BENCHMARK_START_TIMER_SECTION("getDoFs and inject state");
        VecX_T<ADReal> ad_xp = m_linesearch_umbrella.getExtendedDoFsPARL();
        for (size_t i = 0; i < np + nd; ++i) ad_xp[i].derivatives()[0] = delta_xp[i];
        m_diff_umbrella.setExtendedDoFsPARL(ad_xp);
        BENCHMARK_STOP_TIMER_SECTION("getDoFs and inject state");

        BENCHMARK_START_TIMER_SECTION("delta_dJ_dxp");
        delta_dJ_dxp = objective.delta_grad(delta_xp, m_diff_umbrella, opt_eType);
        BENCHMARK_STOP_TIMER_SECTION("delta_dJ_dxp");

        m_delta_w_rhs = delta_dJ_dxp.head(nd); // debugging

        // Solve for adjoint state perturbation
        // H delta_w = [ d^2J/dpdxp delta_xp ] - [d3E/dx dx dxp delta_xp] w
        //             \__________________________________________________/
        //                                     b
        if (coeff_J != 0.0) {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta w x");
            BENCHMARK_START_TIMER_SECTION("Hw");
            VecX_T<ADReal> w_padded(nd + np);
            w_padded.head(nd) = objective.adjointState();
            w_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_w for evaluating the full Hessian matvec expressions below...
            m_d3E_w = extractDirectionalDerivative(m_diff_umbrella.applyHessianPerArmRestlen(w_padded, mask_dxpdx));
            BENCHMARK_STOP_TIMER_SECTION("Hw");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");

            auto b = (delta_dJ_dxp.head(nd) - m_d3E_w.head(nd)).eval();
            m_linesearch_ws->getFreeComponentInPlace(b); // Enforce any active bound constraints

            if (opt.get_problem().hasLEQConstraint()) m_delta_w = opt.extractFullSolution(opt.kkt_solver(opt.solver, opt.removeFixedEntries(b)));
            else                                      m_delta_w = opt.extractFullSolution(          opt.solver.solve(opt.removeFixedEntries(b)));

            m_linesearch_ws->validateStep(m_delta_w); // Debugging: ensure the adjoint state components corresponding to constrained variables remain zero.

            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }
    }
    catch (...) {
        std::cout << "HESSVEC FAIL!!!" << std::endl;
        m_delta_x    = VecX_T<Real>::Zero(nd     );
        m_delta_w    = VecX_T<Real>::Zero(nd     );
        m_d3E_w      = VecX_T<Real>::Zero(nd + np);
        delta_dJ_dxp = VecX_T<Real>::Zero(nd + np);
    }

    VecX_T<Real> result;
    result.setZero(np);
    // Accumulate the J hessian matvec
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer3("evaluate hessian matvec");
        if (coeff_J != 0.0) {
            VecX_T<Real> delta_edofs(nd + np);
            delta_edofs.head(nd) = m_delta_w;
            delta_edofs.tail(np).setZero();

            HessianComputationMask mask;
            mask.dof_out = false;
            mask.designParameter_in = false;

            result += delta_dJ_dxp.tail(np)
                   -  m_linesearch_umbrella.applyHessianPerArmRestlen(delta_edofs, mask).tail(np)
                   -  m_d3E_w.tail(np);
            result *= coeff_J;
        }
    }

    return result;
}

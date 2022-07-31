#if HAS_KNITRO

#include <knitro.hh>

template<class OptimizationObject>
struct OptKnitroProblem : public KnitroProblem<OptKnitroProblem<OptimizationObject>> {
    using Base = KnitroProblem<OptKnitroProblem<OptimizationObject>>;

    OptKnitroProblem(OptimizationObject &um_opt, double minRestLen)
        : Base(um_opt.numParams(), um_opt.numLinearInequalityConstraints()), m_um_opt(um_opt)
    {
        this->setObjType(KPREFIX(OBJTYPE_GENERAL));
        this->setObjGoal(KPREFIX(OBJGOAL_MINIMIZE));

        // Set the bounds for the design parameters:
        //     Rest length parameters are bewteen epsilon and infinity

        if (minRestLen < 0) minRestLen = m_um_opt.defaultLengthBound();
        std::vector<double> restLenLoBounds(m_um_opt.numRestLen(), minRestLen);
        std::vector<double> restLenUpBounds(m_um_opt.numRestLen(), KPREFIX(INFBOUND));

        const size_t nc = um_opt.numLinearInequalityConstraints();
        if (nc > 0) {
#if KNITRO_LEGACY
            throw std::runtime_error("Linear inequality constarints are not implemented yet for legacy Knitro");
#else
            auto lics = um_opt.getLinearInequalityConstraints(minRestLen);
            for (size_t ci = 0; ci < nc; ++ci) {
                this->getConstraintsConstPart()  .add(ci, lics[ci].constPart);
                this->getConstraintsLinearParts().add(ci, knitro::KNLinearStructure(lics[ci].vars, lics[ci].coeffs));
            }
#endif
        }

        this->setVarLoBnds(restLenLoBounds);
        this->setVarUpBnds(restLenUpBounds);
    }

    double evalFC(const double *x,
                        double * /* cval */,
                        double * objGrad,
                        double * /* jac */) {
        const size_t np = m_um_opt.numParams();
        auto params = Eigen::Map<const Eigen::VectorXd>(x, np);
        Real val = m_um_opt.J(params);

        auto g = m_um_opt.gradp_J(params);
        Eigen::Map<Eigen::VectorXd>(objGrad, np) = g;
        return val;
    }

    int evalGA(const double * /* x */, double * /* objGrad */, double * /* jac */) {
        // Tell Knitro that gradient is evaluated by evaluateFC
        return KPREFIX(RC_EVALFCGA);
    }

    // Note: "lambda" contains a Lagrange multiplier for each constraint and each variable.
    // The first numConstraints entries give each constraint's multiplier in order, and the remaining
    // numVars entries give each the multiplier for the variable's active simple bound constraints (if any).
    int evalHessVec(const double *x, double sigma, const double * /* lambda */,
                    const double *vec, double *hessVec) {
        const size_t np = m_um_opt.numParams();
        if (sigma == 0.0) throw std::runtime_error("Knitro requested empty Hessian!");

        auto params  = Eigen::Map<const Eigen::VectorXd>(x, np);
        auto delta_p = Eigen::Map<const Eigen::VectorXd>(vec, np);

        // Apply Hessian of sigma * J + lambda[0] * angle_constraint if angle constraint is active, J otherwise
        auto result = m_um_opt.apply_hess(params, delta_p, sigma);
        Eigen::Map<Eigen::VectorXd>(hessVec, np) = result;
        return 0; // indicate success
    }

private:
    OptimizationObject &m_um_opt;
};

template<class OptimizationObject>
struct OptKnitroNewPtCallback : public NewPtCallbackBase {
    OptKnitroNewPtCallback(OptimizationObject &um_opt, std::function<void()> update_viewer)
        : NewPtCallbackBase(), m_update_viewer(update_viewer), m_um_opt(um_opt) { }

    virtual int operator()(const double *x) override {
        const size_t np = m_um_opt.numParams();
        m_um_opt.newPt(Eigen::Map<const Eigen::VectorXd>(x, np));
        m_update_viewer();
        return 0;
    }
private:
    std::function<void()> m_update_viewer;
    OptimizationObject &m_um_opt;
};

void configureKnitroSolver(KnitroSolver &solver, int num_steps, Real trust_region_scale, Real optimality_tol) {
    solver.useNewptCallback();
    solver.setParam(KPREFIX(PARAM_HONORBNDS), KPREFIX(HONORBNDS_ALWAYS)); // always respect bounds during optimization
    solver.setParam(KPREFIX(PARAM_MAXIT),     num_steps);
    solver.setParam(KPREFIX(PARAM_PRESOLVE),  KPREFIX(PRESOLVE_NONE));
    solver.setParam(KPREFIX(PARAM_DELTA),     trust_region_scale);
    // solver.setParam(KPREFIX(PARAM_DERIVCHECK), KPREFIX(DERIVCHECK_ALL));
    // solver.setParam(KPREFIX(PARAM_DERIVCHECK_TYPE), KPREFIX(DERIVCHECK_CENTRAL));
    // solver.setParam(KPREFIX(PARAM_ALGORITHM), KPREFIX(ALG_BAR_DIRECT));   // interior point with exact Hessian
    solver.setParam(KPREFIX(PARAM_PAR_NUMTHREADS), 12);
    solver.setParam(KPREFIX(PARAM_HESSIAN_NO_F), KPREFIX(HESSIAN_NO_F_ALLOW)); // allow Knitro to call our hessvec with sigma = 0
    // solver.setParam(KPREFIX(PARAM_LINSOLVER), KPREFIX(LINSOLVER_MKLPARDISO));
    solver.setParam(KPREFIX(PARAM_ALGORITHM), KPREFIX(ALG_ACT_CG));
    solver.setParam(KPREFIX(PARAM_ACT_QPALG), KPREFIX(ACT_QPALG_ACT_CG)); // default ended up choosing KPREFIX(ACT_QPALG_BAR_DIRECT)
    // solver.setParam(KPREFIX(PARAM_CG_MAXIT), 25);
    // solver.setParam(KPREFIX(PARAM_CG_MAXIT), int(um_opt.numParams())); // TODO: decide on this.
    // solver.setParam(KPREFIX(PARAM_BAR_FEASIBLE), KPREFIX(BAR_FEASIBLE_NO));

    solver.setParam(KPREFIX(PARAM_OPTTOL), optimality_tol);
    solver.setParam(KPREFIX(PARAM_OUTLEV), KPREFIX(OUTLEV_ALL));
}

template<class OptimizationObject>
int optimize(OptimizationObject &um_opt, OptAlgorithm alg, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, std::function<void()> &update_viewer, double minRestLen) {
    OptKnitroProblem<OptimizationObject> problem(um_opt, minRestLen);

    std::vector<Real> x_init(um_opt.numParams());
    Eigen::Map<Eigen::VectorXd>(x_init.data(), x_init.size()) = um_opt.params();
    problem.setXInitial(x_init);

    OptKnitroNewPtCallback<OptimizationObject> callback(um_opt, update_viewer);
    problem.setNewPointCallback(&callback);
    // Create a solver - optional arguments:
    int hessopt = 0;
    if (alg == OptAlgorithm::NEWTON_CG) hessopt = KPREFIX(HESSOPT_PRODUCT); // exact Hessian-vector products
    else if (alg == OptAlgorithm::BFGS) hessopt = KPREFIX(HESSOPT_BFGS   ); // BFGS approximation
    else throw std::runtime_error("Unknown algorithm");

    KnitroSolver solver(&problem, /* exact gradients */ 1, hessopt);
    configureKnitroSolver(solver, int(num_steps), trust_region_scale, optimality_tol);

    int solveStatus = 1;
    try {
        BENCHMARK_RESET();
        int solveStatus = solver.solve();
        BENCHMARK_REPORT_NO_MESSAGES();

        if (solveStatus != 0) {
            std::cout << std::endl;
            std::cout << "KNITRO failed to solve the problem, final status = ";
            std::cout << solveStatus << std::endl;
        }
    }
    catch (KnitroException &e) {
        problem.setNewPointCallback(nullptr);
        printKnitroException(e);
        throw e;
    }
    problem.setNewPointCallback(nullptr);

    return solveStatus;
}

#else // !HAS_KNITRO

template<class OptimizationObject>
int optimize(OptimizationObject &/* um_opt */, OptAlgorithm /* alg */, size_t /* num_steps */,
              Real /* trust_region_scale */, Real /* optimality_tol */, std::function<void()> &/* update_viewer */, double /* minRestLen */) {
    throw std::runtime_error("Knitro is not available.");
}

#endif

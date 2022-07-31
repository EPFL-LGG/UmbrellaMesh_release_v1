#ifndef DESIGNPARAMETER_SOLVE_HH
#define DESIGNPARAMETER_SOLVE_HH

#include <vector>
#include <cmath>
#include <memory>
#include <map>

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include "compute_equilibrium.hh"


// Solve an equilibrium problem augemented with rest length variables.
template<typename Object>
struct DesignParameterProblem : public NewtonProblem {
    // E0, l0 are the elastic energy/length of the original design.
    // These can be specified manually in case the DesignParameterProblem is being
    // constructed from a design that has already been modified.
    DesignParameterProblem(Object &obj, Real E0 = -1)
        : object(obj), m_characteristicLength(obj.characteristicLength())
    {
        m_E0 = (E0 > 0) ? E0 : obj.designParameterSolve_energy();
        // Make sure length variables aren't shrunk down to zero/inverted
        // when the sign of the length variables flips, the corresponding tangent vector will turn exactly 180 degrees, which is singularity for parallel transport. 
        if (obj.getDesignParameterConfig().restLen) {
            const Real initMinRestLen = obj.initialMinRestLength(); 
            auto lengthVars = obj.designParameterSolve_lengthVars();
            m_boundConstraints.reserve(lengthVars.size());
            for (size_t var : lengthVars)
                m_boundConstraints.emplace_back(var, 0.01 * initMinRestLen, BoundConstraint::Type::LOWER);
        }

        setFixedVars(obj.designParameterSolveFixedVars());

        m_hessianSparsity = obj.designParameterSolve_hessianSparsityPattern();
    }

    virtual void setVars(const Eigen::VectorXd &vars) override { object.designParameterSolve_setDoF(vars); }
    virtual const Eigen::VectorXd getVars() const override { return object.designParameterSolve_getDoF(); }
    virtual size_t numVars() const override { return object.designParameterSolve_numDoF(); }

    virtual Real energy() const override {
        Real result = gamma / m_E0 * object.designParameterSolve_energy();
        return result;
    }

    virtual Eigen::VectorXd gradient(bool freshIterate = false) const override {
        Eigen::VectorXd g = gamma / m_E0 * object.designParameterSolve_gradient(freshIterate, UmbrellaMesh::UmbrellaEnergyType::Full, UmbrellaMesh::EnergyType::Full);
        return g;
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { return m_hessianSparsity; }

    virtual void writeDebugFiles(const std::string &errorName) const override {
        auto H = object.hessian();
        H.rowColRemoval(fixedVars());
        H.reflectUpperTriangle();
        H.dumpBinary("debug_" + errorName + "_hessian.mat");
        object.saveVisualizationGeometry("debug_" + errorName + "_geometry.msh");
    }

    void set_gamma (const Real new_gamma)             { gamma = new_gamma; m_clearCache(); }

    Real weighted_energy() const { return gamma / m_E0 * object.designParameterSolve_energy(); }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

    Real elasticEnergyWeight() const { return gamma / m_E0; }

    Real E0() const { return m_E0; }

private:
    virtual void m_iterationCallback(size_t i) override {
        object.updateSourceFrame(); object.updateRotationParametrizations();
        if (m_customCallback) m_customCallback(*this, i);
    }

    virtual void m_evalHessian(SuiteSparseMatrix &result, bool /* projectionMask */) const override {
        result.setZero();
        object.designParameterSolve_hessian(result, UmbrellaMesh::UmbrellaEnergyType::Full, UmbrellaMesh::EnergyType::Full);
        result.scale(gamma / m_E0);
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        result.setZero();
        object.massMatrix(result);
        const size_t dpo = object.designParameterOffset(), ndp = object.designParameterSolve_numDesignParameters();
        for (size_t j = 0; j < ndp; ++j) {
            result.addNZ(result.findDiagEntry(dpo + j), m_characteristicLength);
            // TODO: figure out a more sensible mass to use for rest length variables.
            // Initial mass of each segment?
        }
    }

    Object &object;
    mutable SuiteSparseMatrix m_hessianSparsity;
    Real m_characteristicLength = 1.0;
    Real m_E0 = 1.0;
    Real gamma = 1;

    CallbackFunction m_customCallback;
};

template<typename Object>
std::unique_ptr<DesignParameterProblem<Object>> designParameter_problem(Object &obj, const std::vector<size_t> &fixedVars = std::vector<size_t>(), Real E0 = -1) {
    auto problem = std::make_unique<DesignParameterProblem<Object>>(obj, E0);
    // Also fix the variables specified by the user.
    problem->addFixedVariables(fixedVars);
    return problem;
}

template<typename Object>
std::unique_ptr<NewtonOptimizer> get_designParameter_optimizer(Object &obj, const std::vector<size_t> &fixedVars = std::vector<size_t>(), CallbackFunction customCallback = nullptr, Real E0 = -1) {
    auto problem = designParameter_problem(obj, fixedVars, E0);
    problem->setCustomIterationCallback(customCallback);
    return std::make_unique<NewtonOptimizer>(std::move(problem));
}

// Rest length solve with custom optimizer options.
template<typename Object>
ConvergenceReport designParameter_solve(Object &obj, const NewtonOptimizerOptions &opts, const std::vector<size_t> &fixedVars = std::vector<size_t>(), CallbackFunction customCallback = nullptr, Real E0 = -1) {
    auto opt = get_designParameter_optimizer(obj, fixedVars, customCallback, E0);
    opt->options = opts;
    return opt->optimize();
}

// Default options for rest length solve: use the identity metric.
template<typename Object>
ConvergenceReport designParameter_solve(Object &obj, const std::vector<size_t> &fixedVars = std::vector<size_t>(), CallbackFunction customCallback = nullptr, Real E0 = -1) {
    NewtonOptimizerOptions opts;
    opts.useIdentityMetric = true;
    return designParameter_solve(obj, opts, fixedVars, customCallback, E0);
}

#endif /* end of include guard: DESIGNPARAMETER_SOLVE_HH */

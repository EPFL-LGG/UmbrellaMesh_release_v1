#include <MeshFEM/Geometry.hh>

#include "../UmbrellaMesh.hh"
#include "../UmbrellaDesignOptimizationTerms.hh"
#include "../UmbrellaOptimization.hh"
#include "../UmbrellaRestHeightsOptimization.hh"
#include "../UmbrellaSingleRestHeightOptimization.hh"
#include "../UmbrellaFourParametersOptimization.hh"
#include "../UmbrellaKnitroProblem.hh"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
#include <sstream>
namespace py = pybind11;


using UM = UmbrellaMesh_T<Real>;


template<typename T>
std::string hexString(T val) {
    std::ostringstream ss;
    ss << std::hex << val;
    return ss.str();
}

PYBIND11_MODULE(umbrella_optimization, m) {
    py::module::import("MeshFEM");
    py::module::import("py_newton_optimizer");
    py::module::import("umbrella_mesh");
    m.doc() = "Umbrella Mesh Optimization Codebase";

    py::module detail_module = m.def_submodule("detail");

    using OEType = OptEnergyType;

    py::enum_<OEType>(m, "OptEnergyType")
    .value("Full",            OEType::Full           )
    .value("Elastic",         OEType::Elastic        )
    .value("Target",          OEType::Target         )
    .value("DeploymentForce", OEType::DeploymentForce)
    .value("Stress",          OEType::Stress         )
    .value("UmbrellaForces",  OEType::UmbrellaForces )
    ;

    using DOT = DesignOptimizationTerm<UmbrellaMesh_T>;
    py::class_<DOT, std::shared_ptr<DOT>>(m, "DesignOptimizationTerm")
        .def("value",  &DOT::value)
        .def("update", &DOT::update)
        .def("grad"  , &DOT::grad  )
        .def("grad_x", &DOT::grad_x)
        .def("grad_p", &DOT::grad_p)
        .def("object",           &DOT::object, py::return_value_policy::reference)
        .def("computeGrad",      &DOT::computeGrad)
        .def("computeDeltaGrad", &DOT::computeDeltaGrad, py::arg("delta_xp"))
        .def("numVars", &DOT::numVars)
        .def("numSimVars", &DOT::numSimVars)
        .def("numDesignVars", &DOT::numDesignVars)
        ;

    using DOOT = DesignOptimizationObjectiveTerm<UmbrellaMesh_T>;
    py::class_<DOOT, DOT, std::shared_ptr<DOOT>>(m, "DesignOptimizationObjectiveTerm")
        .def_readwrite("weight", &DOOT::weight)
        ;

    using EEO = ElasticEnergyObjective<UmbrellaMesh_T>;
    py::class_<EEO, DOOT, std::shared_ptr<EEO>>(m, "ElasticEnergyObjective")
        .def(py::init<const UM &>(), py::arg("umbrella_mesh"))
        .def_property("useEnvelopeTheorem", &EEO::useEnvelopeTheorem, &EEO::setUseEnvelopeTheorem)
        ;

    using TFO = TargetFittingDOOT<UmbrellaMesh_T>;
    py::class_<TFO, DOOT, std::shared_ptr<TFO>>(m, "TargetFittingDOOT")
        .def(py::init<const UM &, TargetSurfaceFitter &>(), py::arg("umbrella_mesh"), py::arg("targetSurfaceFitter"))
        ;

    using LSO = LpStressDOOT<UmbrellaMesh_T>;
    py::class_<LSO, DOOT, std::shared_ptr<LSO>>(m, "LpStressDOOT")
        .def(py::init<const UM &>(), py::arg("umbrella_mesh"))
        .def_readwrite("p", &LSO::p)
        .def_readwrite("stressType", &LSO::stressType)
        ;

    using DFO = DeploymentForceDOOT<UmbrellaMesh_T>;
    py::class_<DFO, DOOT, std::shared_ptr<DFO>>(m, "DeploymentForceDOOT")
        .def(py::init<const UM &>(), py::arg("umbrella_mesh"))
        .def_property("activationThreshold", &DFO::getActivationThreshold, &DFO::setActivationThreshold)
        ;

    using UFO = UmbrellaForceObjective<UmbrellaMesh_T>;
    py::class_<UFO, DOOT, std::shared_ptr<UFO>>(m, "UmbrellaForceObjective")
        .def(py::init<const UM &>(), py::arg("umbrella_mesh"))
        .def_property("normalActivationThreshold", &UFO::getNormalActivationThreshold, &UFO::setNormalActivationThreshold)
        .def_property(             "normalWeight", &UFO::getNormalWeight,              &UFO::setNormalWeight)
        .def_property(         "tangentialWeight", &UFO::getTangentialWeight,          &UFO::setTangentialWeight)
        .def_property(             "torqueWeight", &UFO::getTorqueWeight,              &UFO::setTorqueWeight)
        ;
    ////////////////////////////////////////////////////////////////////////////////
    // Individual rod Lp stress objective (debugging)
    ////////////////////////////////////////////////////////////////////////////////
    using DOTR = DesignOptimizationTerm<ElasticRod_T>;
    py::class_<DOTR, std::shared_ptr<DOTR>>(m, "DesignOptimizationTermRod")
        .def("value",  &DOTR::value)
        .def("update", &DOTR::update)
        .def("grad"  , &DOTR::grad  )
        .def("grad_x", &DOTR::grad_x)
        .def("grad_p", &DOTR::grad_p)
        .def("object",           &DOTR::object, py::return_value_policy::reference)
        .def("computeGrad",      &DOTR::computeGrad)
        .def("computeDeltaGrad", &DOTR::computeDeltaGrad, py::arg("delta_xp"))
        .def("numVars",       &DOTR::numVars)
        .def("numSimVars",    &DOTR::numSimVars)
        .def("numDesignVars", &DOTR::numDesignVars)
        ;
    using DOOTR = DesignOptimizationObjectiveTerm<ElasticRod_T>;
    py::class_<DOOTR, DOTR, std::shared_ptr<DOOTR>>(m, "DesignOptimizationObjectiveTermRod")
        .def_readwrite("weight", &DOOTR::weight)
        ;
    using LSOR = LpStressDOOT<ElasticRod_T>;
    py::class_<LSOR, DOOTR, std::shared_ptr<LSOR>>(m, "LpStressDOOTR")
        .def(py::init<const ElasticRod_T<Real> &>(), py::arg("rod"))
        .def_readwrite("p", &LSOR::p)
        .def_readwrite("stressType", &LSOR::stressType)
        ;
    ////////////////////////////////////////////////////////////////////////////////
    // End individual rod Lp stress objective (debugging)
    ////////////////////////////////////////////////////////////////////////////////

    using DOO  = DesignOptimizationObjective<UmbrellaMesh_T, OEType>;
    using TR   = DOO::TermRecord;
    using TPtr = DOO::TermPtr;
    py::class_<DOO> doo(m, "DesignOptimizationObjective");

    py::class_<TR>(doo, "DesignOptimizationObjectiveTermRecord")
        .def(py::init<const std::string &, OEType, std::shared_ptr<DOT>>(),  py::arg("name"), py::arg("type"), py::arg("term"))
        .def_readwrite("name", &TR::name)
        .def_readwrite("type", &TR::type)
        .def_readwrite("term", &TR::term)
        .def("__repr__", [](const TR *trp) {
                const auto &tr = *trp;
                return "TermRecord " + tr.name + " at " + hexString(trp) + " with weight " + std::to_string(tr.term->getWeight()) + " and value " + std::to_string(tr.term->unweightedValue());
        })
        ;

    doo.def(py::init<>())
       .def("update",         &DOO::update)
       .def("grad",           &DOO::grad, py::arg("type") = OEType::Full)
       .def("values",         &DOO::values)
       .def("weightedValues", &DOO::weightedValues)
       .def("value",  py::overload_cast<OEType>(&DOO::value, py::const_), py::arg("type") = OEType::Full)
        .def("computeGrad",     &DOO::computeGrad, py::arg("type") = OEType::Full)
       .def("computeDeltaGrad", &DOO::computeDeltaGrad, py::arg("delta_xp"), py::arg("type") = OEType::Full)
       .def_readwrite("terms",  &DOO::terms, py::return_value_policy::reference)
       .def("add", py::overload_cast<const std::string &, OEType, TPtr      >(&DOO::add),  py::arg("name"), py::arg("type"), py::arg("term"))
       .def("add", py::overload_cast<const std::string &, OEType, TPtr, Real>(&DOO::add),  py::arg("name"), py::arg("type"), py::arg("term"), py::arg("weight"))
       // More convenient interface for adding multiple terms at once
       .def("add", [](DOO &o, const std::list<std::tuple<std::string, OEType, TPtr>> &terms) {
                    for (const auto &t : terms)
                        o.add(std::get<0>(t), std::get<1>(t), std::get<2>(t));
               })
       ;

    ////////////////////////////////////////////////////////////////////////////////
    // Umbrella PARL Optimization
    ////////////////////////////////////////////////////////////////////////////////

    py::enum_<OptAlgorithm>(m, "OptAlgorithm")
        .value("NEWTON_CG", OptAlgorithm::NEWTON_CG)
        .value("BFGS",      OptAlgorithm::BFGS     )
        ;

    using UO = UmbrellaOptimization;
    auto umbrella_parl_optimization = py::class_<UO>(m, "UmbrellaOptimization")
        .def(py::init<UmbrellaMesh &, const NewtonOptimizerOptions &, const Real, int, bool, const std::vector<size_t> &>(), py::arg("umbrella_mesh"), py::arg("eopts"), py::arg("elasticEnergyIncreaseFactorLimit"), py::arg("pinJoint"), py::arg("useFixedJoint"), py::arg("fixedVars"))
        .def_property("beta",  &UO::getBeta , &UO::setBeta )
        .def_property("gamma", &UO::getGamma, &UO::setGamma)
        .def_property("eta",   &UO::getEta,   &UO::setEta)
        .def_property("zeta",  &UO::getZeta,  &UO::setZeta)
        .def_property("iota",  &UO::getIota,  &UO::setIota)
        .def("numParams", &UO::numParams)
        .def("newPt",     &UO::newPt, py::arg("params"))
        .def("J",         py::overload_cast<>(&UO::J))
        .def("J",         py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&UO::J), py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
        .def("J_target",  py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&UO::J_target), py::arg("params"))
        .def("gradp_J",   py::overload_cast<>(&UO::gradp_J))
        .def("gradp_J",   py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&UO::gradp_J), py::arg("params"), py::arg("energyType") = OptEnergyType::Full)

        .def("commitLinesearchUmbrella", &UO::commitLinesearchUmbrella)
        .def("invalidateEquilibria",     &UO::invalidateEquilibria)
        .def("invalidateAdjointState",   &UO::invalidateAdjointState)

        .def("apply_hess",   &UO::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J"), py::arg("energyType") = OptEnergyType::Full)   
        .def("apply_hess_J", &UO::apply_hess_J, py::arg("params"), py::arg("delta_p"), py::arg("energyType") = OptEnergyType::Full)   
        .def("params",       &UO::params)
        .def("setHoldClosestPointsFixed", &UO::setHoldClosestPointsFixed, py::arg("attractionHCP"), py::arg("objectiveHCP"))
        .def("reset_joint_target_with_closest_points", &UO::reset_joint_target_with_closest_points)
        .def("setAttractionWeight",  &UO::setAttractionWeight, py::arg("attraction_weight"))
        .def("getAttractionWeight",  &UO::getAttractionWeight)
        .def_property_readonly("equilibriumOptimizer", &UO::getEquilibriumOptimizer, py::return_value_policy::reference)
        .def_property_readonly(    "linesearchObject", &UO::       linesearchObject, py::return_value_policy::reference)
        .def_property_readonly(     "committedObject", &UO::        committedObject, py::return_value_policy::reference)
        .def_property_readonly("linesearchWorkingSet", &UO::   linesearchWorkingSet, py::return_value_policy::reference)
        .def_property_readonly( "committedWorkingSet", &UO::    committedWorkingSet, py::return_value_policy::reference)
        .def_property("prediction_order", [](const UO &uo) { return int(uo.prediction_order); },
                                          [](      UO &uo, unsigned order) { if (order > 2) throw std::runtime_error("Unsupported"); uo.prediction_order = PredictionOrder(order); })
        .def_property_readonly("delta_x", &UO::get_delta_x)
        .def_property_readonly("w",       &UO::get_w)
        .def_property_readonly("delta_w", &UO::get_delta_w)
        .def_property_readonly("d3E_w",   &UO::get_d3E_w)
        .def_property_readonly("w_rhs",   &UO::get_w_rhs)
        .def_property_readonly("delta_w_rhs", &UO::get_delta_w_rhs)
        .def("get_H_times", &UO::get_H_times)

        .def_readwrite("objective", &UO::objective, py::return_value_policy::reference)
        .def_readonly("target_surface_fitter", &UO::target_surface_fitter)
        .def("defaultLengthBound", &UO::defaultLengthBound)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Umbrella Rest Heights Optimization
    ////////////////////////////////////////////////////////////////////////////////
    using URHO = UmbrellaRestHeightsOptimization;
    auto umbrella_rest_heights_optimization = py::class_<URHO>(m, "UmbrellaRestHeightsOptimization")
        .def(py::init<UmbrellaOptimization &>(), py::arg("umbrella_optimization"))
        .def("numParams",      &URHO::numParams)
        .def("params",         &URHO::params)
        .def("newPt",          &URHO::newPt, py::arg("params"))
        .def("get_parent_opt", &URHO::get_parent_opt)
        .def_property("beta",  &URHO::getBeta , &URHO::setBeta )
        .def_property("gamma", &URHO::getGamma, &URHO::setGamma)
        .def_property("eta",   &URHO::getEta,   &URHO::setEta)
        .def_property("zeta",  &URHO::getZeta,  &URHO::setZeta)
        .def_property("iota",  &URHO::getIota,  &URHO::setIota)
        .def("J",              py::overload_cast<>(&URHO::J))
        .def("J",              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&URHO::J),              py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
        .def("gradp_J",        py::overload_cast<>(&URHO::gradp_J))
        .def("gradp_J",        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&URHO::gradp_J),        py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
        .def("apply_hess",     &URHO::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J"), py::arg("energyType") = OptEnergyType::Full)   
        .def("apply_hess_J",   &URHO::apply_hess_J, py::arg("params"), py::arg("delta_p"), py::arg("energyType") = OptEnergyType::Full)
        .def("applyTransformation", &URHO::applyTransformation, py::arg("URH"))
        .def("applyTransformationTranspose", &URHO::applyTransformationTranspose, py::arg("PARL"))
        .def("reset_joint_target_with_closest_points", &URHO::reset_joint_target_with_closest_points)
        .def("invalidateAdjointState", &URHO::invalidateAdjointState)
        .def_readwrite("objective", &URHO::objective, py::return_value_policy::reference)
        .def_property_readonly(    "linesearchObject", &URHO::    linesearchObject, py::return_value_policy::reference)
        .def_property_readonly(     "committedObject", &URHO::     committedObject, py::return_value_policy::reference)
        .def("defaultLengthBound", &URHO::defaultLengthBound)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Umbrella Single Rest Height Optimization
    ////////////////////////////////////////////////////////////////////////////////
    using USRHO = UmbrellaSingleRestHeightOptimization;
    auto umbrella_single_rest_height_optimization = py::class_<USRHO>(m, "UmbrellaSingleRestHeightOptimization")
        .def(py::init<UmbrellaRestHeightsOptimization &>(), py::arg("umbrella_rest_heights_optimization"))
        .def("numParams",      &USRHO::numParams)
        .def("params",         &USRHO::params)
        .def("newPt",          &USRHO::newPt, py::arg("params"))
        .def("get_parent_opt", &USRHO::get_parent_opt)
        .def_property("beta",  &USRHO::getBeta , &USRHO::setBeta )
        .def_property("gamma", &USRHO::getGamma, &USRHO::setGamma)
        .def_property("eta",   &USRHO::getEta,   &USRHO::setEta)
        .def_property("zeta",  &USRHO::getZeta,  &USRHO::setZeta)
        .def_property("iota",  &USRHO::getIota,  &USRHO::setIota)
        .def("J",              py::overload_cast<>(&USRHO::J))
        .def("J",              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&USRHO::J),             py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
        .def("gradp_J",        py::overload_cast<>(&USRHO::gradp_J))
        .def("gradp_J",        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&USRHO::gradp_J),       py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
        .def("apply_hess",     &USRHO::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J"), py::arg("energyType") = OptEnergyType::Full)   
        .def("apply_hess_J",   &USRHO::apply_hess_J, py::arg("params"), py::arg("delta_p"), py::arg("energyType") = OptEnergyType::Full)
        .def("applyTransformation", &USRHO::applyTransformation, py::arg("URH"))
        .def("applyTransformationTranspose", &USRHO::applyTransformationTranspose, py::arg("PARL") )
        .def("invalidateAdjointState", &USRHO::invalidateAdjointState)
        .def("reset_joint_target_with_closest_points", &USRHO::reset_joint_target_with_closest_points)
        .def_readwrite("objective", &USRHO::objective, py::return_value_policy::reference)
        .def_property_readonly(    "linesearchObject", &USRHO::    linesearchObject, py::return_value_policy::reference)
        .def_property_readonly(     "committedObject", &USRHO::     committedObject, py::return_value_policy::reference)
        .def("defaultLengthBound", &USRHO::defaultLengthBound)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Umbrella Four Parameters Optimization
    ////////////////////////////////////////////////////////////////////////////////
    using URFP = UmbrellaFourParametersOptimization;
    auto umbrella_four_parameters_optimization = py::class_<URFP>(m, "UmbrellaFourParametersOptimization")
        .def(py::init<UmbrellaOptimization &>(), py::arg("umbrella_optimization"))
        .def("numParams",      &URFP::numParams)
        .def("params",         &URFP::params)
        .def("newPt",          &URFP::newPt, py::arg("params"))
        .def("get_parent_opt", &URFP::get_parent_opt)
        .def_property("beta",  &URFP::getBeta , &URFP::setBeta )
        .def_property("gamma", &URFP::getGamma, &URFP::setGamma)
        .def_property("eta",   &URFP::getEta,   &URFP::setEta)
        .def_property("zeta",  &URFP::getZeta,  &URFP::setZeta)
        .def_property("iota",  &URFP::getIota,  &URFP::setIota)
        .def("J",              py::overload_cast<>(&URFP::J))
        .def("J",              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&URFP::J),              py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
        .def("gradp_J",        py::overload_cast<>(&URFP::gradp_J))
        .def("gradp_J",        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&URFP::gradp_J),        py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
        .def("apply_hess",     &URFP::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J"), py::arg("energyType") = OptEnergyType::Full)   
        .def("apply_hess_J",   &URFP::apply_hess_J, py::arg("params"), py::arg("delta_p"), py::arg("energyType") = OptEnergyType::Full)
        .def("applyTransformation", &URFP::applyTransformation, py::arg("URH"))
        .def("applyTransformationTranspose", &URFP::applyTransformationTranspose, py::arg("PARL"))
        .def("invalidateAdjointState", &URFP::invalidateAdjointState)
        .def("reset_joint_target_with_closest_points", &URFP::reset_joint_target_with_closest_points)
        .def_readwrite("objective", &URFP::objective, py::return_value_policy::reference)
        .def_property_readonly(    "linesearchObject", &URFP::    linesearchObject, py::return_value_policy::reference)
        .def_property_readonly(     "committedObject", &URFP::     committedObject, py::return_value_policy::reference)
        .def("defaultLengthBound", &URFP::defaultLengthBound)
        ;
    ////////////////////////////////////////////////////////////////////////////////
    // Knitro Optimization
    ////////////////////////////////////////////////////////////////////////////////
#if HAS_KNITRO

    m.def("optimize",
        [](UO &um_opt, OptAlgorithm alg, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, std::function<void()> &update_viewer, double minRestLen) {
            return optimize(um_opt, alg, num_steps, trust_region_scale, optimality_tol, update_viewer, minRestLen);
        },
        py::arg("umbrella_optimization"), 
        py::arg("alg"), 
        py::arg("num_steps"), 
        py::arg("trust_region_scale"), 
        py::arg("optimality_tol"), 
        py::arg("update_viewer"), 
        py::arg("minRestLen") = -1
    );

    m.def("optimize",
        [](URHO &um_opt, OptAlgorithm alg, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, std::function<void()> &update_viewer, double minRestLen) {
            return optimize(um_opt, alg, num_steps, trust_region_scale, optimality_tol, update_viewer, minRestLen);
        },
        py::arg("umbrella_rest_heights_optimization"), 
        py::arg("alg"), 
        py::arg("num_steps"), 
        py::arg("trust_region_scale"), 
        py::arg("optimality_tol"), 
        py::arg("update_viewer"), 
        py::arg("minRestLen") = -1
    );

    m.def("optimize",
        [](USRHO &um_opt, OptAlgorithm alg, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, std::function<void()> &update_viewer, double minRestLen) {
            return optimize(um_opt, alg, num_steps, trust_region_scale, optimality_tol, update_viewer, minRestLen);
        },
        py::arg("umbrella_rest_heights_optimization"), 
        py::arg("alg"), 
        py::arg("num_steps"), 
        py::arg("trust_region_scale"), 
        py::arg("optimality_tol"), 
        py::arg("update_viewer"), 
        py::arg("minRestLen") = -1
    );

    m.def("optimize",
        [](URFP &um_opt, OptAlgorithm alg, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, std::function<void()> &update_viewer, double minRestLen) {
            return optimize(um_opt, alg, num_steps, trust_region_scale, optimality_tol, update_viewer, minRestLen);
        },
        py::arg("umbrella_four_parameters_optimization"), 
        py::arg("alg"), 
        py::arg("num_steps"), 
        py::arg("trust_region_scale"), 
        py::arg("optimality_tol"), 
        py::arg("update_viewer"), 
        py::arg("minRestLen") = -1
    );
#endif

    ////////////////////////////////////////////////////////////////////////////////
    // Benchmarking
    ////////////////////////////////////////////////////////////////////////////////
    m.def("benchmark_reset", &BENCHMARK_RESET);
    m.def("benchmark_start_timer_section", &BENCHMARK_START_TIMER_SECTION, py::arg("name"));
    m.def("benchmark_stop_timer_section",  &BENCHMARK_STOP_TIMER_SECTION,  py::arg("name"));
    m.def("benchmark_start_timer",         &BENCHMARK_START_TIMER,         py::arg("name"));
    m.def("benchmark_stop_timer",          &BENCHMARK_STOP_TIMER,          py::arg("name"));
    m.def("benchmark_report", [](bool includeMessages) {
            py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            if (includeMessages) BENCHMARK_REPORT(); else BENCHMARK_REPORT_NO_MESSAGES();
        },
        py::arg("include_messages") = false)
        ;
}

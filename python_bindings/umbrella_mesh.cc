#include "../UmbrellaMesh.hh"
#include "../UmbrellaMeshTerminalEdgeSensitivity.hh"
#include "../umbrella_compute_equilibrium.hh"
#include "../design_parameter_solve.hh"
#include "../UmbrellaTargetSurfaceFitter.hh"
#include "../LinearActuator.hh"

#include "../3rdparty/elastic_rods/python_bindings/visualization.hh"
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/../../python_bindings/BindingUtils.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// Conversion of std::tuple to and from a py::tuple, since pybind11 doesn't seem to provide this...
template<typename... Args, size_t... Idxs>
py::tuple to_pytuple_helper(const std::tuple<Args...> &args, std::index_sequence<Idxs...>) {
    return py::make_tuple(std::get<Idxs>(args)...);
}

template<typename... Args>
py::tuple to_pytuple(const std::tuple<Args...> &args) {
    return to_pytuple_helper(args, std::make_index_sequence<sizeof...(Args)>());
}

template<class OutType>
struct FromPytupleImpl;

template<typename... Args>
struct FromPytupleImpl<std::tuple<Args...>> {
    template<size_t... Idxs>
    static auto run_helper(const py::tuple &t, std::index_sequence<Idxs...>) {
        return std::make_tuple((t[Idxs].cast<Args>())...);
    }
    static auto run(const py::tuple &t) {
        if (t.size() != sizeof...(Args)) throw std::runtime_error("Mismatched tuple size for py::tuple to std::tuple conversion.");
        return run_helper(t, std::make_index_sequence<sizeof...(Args)>());
    }
};

template<class OutType>
OutType from_pytuple(const py::tuple &t) {
    return FromPytupleImpl<OutType>::run(t);
}

using UMTE = UmbrellaMeshTerminalEdgeSensitivity<Real>;

// Hack around a limitation of pybind11 where we cannot specify argument passing policies and
// pybind11 tries to make a copy if the passed instance is not already registered:
//      https://github.com/pybind/pybind11/issues/1200
// We therefore make our Python callback interface use a raw pointer to forbid this copy (which
// causes an error since NewtonProblem is not copyable).
using PyCallbackFunction = std::function<void(NewtonProblem *, size_t)>;
CallbackFunction callbackWrapper(const PyCallbackFunction &pcb) {
    return [pcb](NewtonProblem &p, size_t i) -> void { if (pcb) pcb(&p, i); };
}

template<typename Object>
void bindDesignParameterProblem(py::module &m, const std::string &typestr) {
    using DPP = DesignParameterProblem<Object>;
    std::string pyclass_name = std::string("DesignParameterProblem_") + typestr;
    py::class_<DPP, NewtonProblem>(m, pyclass_name.c_str())
    .def(py::init<Object &>())
    .def("set_gamma",                  &DPP::set_gamma,                 py::arg("new_gamma"))
    .def("elasticEnergyWeight",        &DPP::elasticEnergyWeight)
    .def("setCustomIterationCallback", [](DPP &dpp, const PyCallbackFunction &pcb) { dpp.setCustomIterationCallback(callbackWrapper(pcb)); })
    .def("weighted_energy",            &DPP::weighted_energy)
    .def("numVars", &DPP::numVars)
    .def("getVars", &DPP::getVars)
    .def("setVars", &DPP::setVars, py::arg("vars"))
    ;

}

PYBIND11_MODULE(umbrella_mesh, m) {
    m.doc() = "Umbrella Mesh Codebase";

    py::module::import("ElasticRods");
    py::module detail_module = m.def_submodule("detail");

    using TSFSD = TargetSurfaceFitter::SurfaceData;
    py::class_<TSFSD, std::shared_ptr<TSFSD>>(m, "TargetSurfaceFitterSurfaceData")
        .def(py::pickle(&TSFSD::serialize, &TSFSD::deserialize))
        ;

    using TSF = TargetSurfaceFitter;
    py::class_<TSF, std::shared_ptr<TSF>>(m, "TargetSurfaceFitter")
        .def(py::init<>())
        .def("loadTargetSurface", &TSF::loadTargetSurface<Real>, py::arg("umbrella"), py::arg("path"))
        .def("objective",         &TSF::objective<Real>,         py::arg("umbrella"))
        .def("gradient",          &TSF::gradient<Real>,          py::arg("umbrella"))
        .def("numQueryPt",        &TSF::numQueryPt<Real>,        py::arg("umbrella"))
        .def("forceUpdateClosestPoints",        &TSF::forceUpdateClosestPoints<Real>,        py::arg("umbrella"))
        .def("getQueryPtPos",     py::overload_cast<const UmbrellaMesh &>(&TSF::getQueryPtPos<Real>, py::const_), py::arg("umbrella"))
        .def("scaleJointWeights", &TSF::scaleJointWeights<Real>, py::arg("umbrella"), py::arg("jointPosWeight"), py::arg("bdryMultiplier") = 1.0, py::arg("featureMultiplier") = 1.0, py::arg("additional_feature_pts") = std::vector<size_t>(), py::arg("updateClosestPoints") = true)

        .def_readwrite("holdClosestPointsFixed", &TSF::holdClosestPointsFixed)

        .def_readonly("W_diag_joint_pos",               &TSF::W_diag_joint_pos)
        .def_readonly("Wsurf_diag_umbrella_sample_pos", &TSF::Wsurf_diag_umbrella_sample_pos)
        .def_readonly("query_pt_pos_tgt",                  &TSF::query_pt_pos_tgt)

        .def_property_readonly("V", [](const TSF &tsf) { return tsf.getV(); })
        .def_property_readonly("F", [](const TSF &tsf) { return tsf.getF(); })
        .def_property_readonly("N", [](const TSF &tsf) { return tsf.getN(); })
        .def_readonly("umbrella_closest_surf_pts",              &TSF::umbrella_closest_surf_pts)
        .def_readonly("umbrella_closest_surf_pt_sensitivities", &TSF::umbrella_closest_surf_pt_sensitivities)
        .def_readonly("umbrella_closest_surf_tris",             &TSF::umbrella_closest_surf_tris)
        .def(py::pickle(&TSF::serialize, &TSF::deserialize))
        ;

    // Linear Actuator
    using LA = LinearActuators<UmbrellaMesh>;
    py::class_<LA>(m, "LinearActuators")
        .def("energy", py::overload_cast<const UmbrellaMesh &, const Eigen::VectorXd &>(&LA::energy, py::const_), py::arg("umbrella"), py::arg("weight"))
        .def_readwrite("angleStiffness", &LA::angleStiffness)
        .def_readwrite("tangentialStiffness", &LA::tangentialStiffness)
        .def_readwrite("axialStiffness", &LA::axialStiffness)
        .def(py::pickle([](const LA &la) { return la.getState(); },
                        [](const LA::SerializedState &t) {
                            return std::make_unique<LA>(t);
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // UmbrellaMesh and nested classes
    ////////////////////////////////////////////////////////////////////////////////

    using PyUM = py::class_<UmbrellaMesh>;
    auto umbrella_mesh = PyUM(m, "UmbrellaMesh");

    py::enum_<UmbrellaMesh::UmbrellaEnergyType>(m, "UmbrellaEnergyType")
        .value("Full",              UmbrellaMesh::UmbrellaEnergyType::Full)
        .value("Elastic",           UmbrellaMesh::UmbrellaEnergyType::Elastic)
        .value("Deployment",        UmbrellaMesh::UmbrellaEnergyType::Deployment)
        .value("Repulsion",         UmbrellaMesh::UmbrellaEnergyType::Repulsion)
        .value("Attraction",        UmbrellaMesh::UmbrellaEnergyType::Attraction)
        .value("AngleBoundPenalty", UmbrellaMesh::UmbrellaEnergyType::AngleBoundPenalty)
        ;
    py::enum_<UmbrellaMesh::DeploymentForceType>(m, "DeploymentForceType")
        .value("Spring",         UmbrellaMesh::DeploymentForceType::Spring)
        .value("Constant",       UmbrellaMesh::DeploymentForceType::Constant)
        .value("LinearActuator", UmbrellaMesh::DeploymentForceType::LinearActuator)
        ;

    py::enum_<UmbrellaMesh::AngleBoundEnforcement>(m, "AngleBoundEnforcement")
        .value("Disable", UmbrellaMesh::AngleBoundEnforcement::Disable)
        .value("Hard",    UmbrellaMesh::AngleBoundEnforcement::Hard)
        .value("Penalty", UmbrellaMesh::AngleBoundEnforcement::Penalty)
        ;

    py::enum_<UmbrellaMesh::JointType>(m, "JointType")
        .value("Rigid", UmbrellaMesh::JointType::Rigid)
        .value("X",     UmbrellaMesh::JointType::X)
        .value("T",     UmbrellaMesh::JointType::T)
        ;

    py::enum_<UmbrellaMesh::JointPosType>(m, "JointPosType")
        .value("Top",   UmbrellaMesh::JointPosType::Top)
        .value("Bot",   UmbrellaMesh::JointPosType::Bot)
        .value("Arm",   UmbrellaMesh::JointPosType::Arm)
        ;

    py::enum_<UmbrellaMesh::SegmentType>(m, "SegmentType")
        .value("Plate",   UmbrellaMesh::SegmentType::Plate)
        .value("Arm",   UmbrellaMesh::SegmentType::Arm)
        ;

    py::class_<UmbrellaMesh::Joint>(umbrella_mesh, "Joint")
        .def("numDoF",     &UmbrellaMesh::Joint::numDoF)
        .def("numBaseDoF", &UmbrellaMesh::Joint::numBaseDoF)
        .def("get_ghost_source_frame",      &UmbrellaMesh::Joint::get_ghost_source_frame)
        .def("get_minus_alpha_sensitivity", &UmbrellaMesh::Joint::get_minus_alpha_sensitivity)
        .def("get_plus_alpha_sensitivity",  &UmbrellaMesh::Joint::get_plus_alpha_sensitivity)
        .def("get_ghost_alpha_sensitivity_frame_A", &UmbrellaMesh::Joint::get_ghost_alpha_sensitivity_frame_A)
        .def("get_ghost_alpha_sensitivity_frame_B", &UmbrellaMesh::Joint::get_ghost_alpha_sensitivity_frame_B)
        .def("input_t", &UmbrellaMesh::Joint::input_t, py::arg("idx"))
        .def("input_p", &UmbrellaMesh::Joint::input_p, py::arg("idx"))
        .def("omega", &UmbrellaMesh::Joint::omega)
        .def("ghost_normal", &UmbrellaMesh::Joint::ghost_normal)
        .def("ghost_source_normal", &UmbrellaMesh::Joint::ghost_source_normal)
        .def_property("alpha",    [](const UmbrellaMesh::Joint &j) { return j.alpha(); }, [](UmbrellaMesh::Joint &j,            Real a) { j.set_alpha(a); })
        .def_property("position", [](const UmbrellaMesh::Joint &j) { return j.pos  (); }, [](UmbrellaMesh::Joint &j, const Vector3D &v) { j.set_pos  (v); })
        .def("getSegmentAt", &UmbrellaMesh::Joint::getSegmentAt, py::arg("localSegmentIndex"))
        .def("getIsStartAt", &UmbrellaMesh::Joint::getIsStartAt, py::arg("localSegmentIndex"))
        .def("terminalEdgeIdentification", &UmbrellaMesh::Joint::terminalEdgeIdentification, py::arg("segmentIndex"))
        .def("jointType", &UmbrellaMesh::Joint::jointType)
        .def("jointPosType", &UmbrellaMesh::Joint::jointPosType)
        .def("numArms", &UmbrellaMesh::Joint::numArms)
        .def("valence", &UmbrellaMesh::Joint::valence)
        .def("umbrellaID", &UmbrellaMesh::Joint::umbrellaID)
        .def(py::pickle([](const UmbrellaMesh::Joint &joint) { return joint.getState(); },
                        [](const UmbrellaMesh::Joint::SerializedState &t) {
                            return std::make_unique<UmbrellaMesh::Joint>(t);
                        }))
        ;

    py::class_<UmbrellaMesh::RodSegment>(umbrella_mesh, "RodSegment")
        .def_readonly("rod",        &UmbrellaMesh::RodSegment::rod, py::return_value_policy::reference)
        .def_readonly("startJoint", &UmbrellaMesh::RodSegment::startJoint)
        .def_readonly("endJoint",   &UmbrellaMesh::RodSegment::endJoint)

        .def("numJoints",           &UmbrellaMesh::RodSegment::numJoints)
        .def("hasStartJoint",       &UmbrellaMesh::RodSegment::hasStartJoint)
        .def("hasEndJoint",         &UmbrellaMesh::RodSegment::hasEndJoint)
        .def("segmentType",         &UmbrellaMesh::RodSegment::segmentType)

        .def("numDoF",              &UmbrellaMesh::RodSegment::numDoF)
        .def("numPosDoF",           &UmbrellaMesh::RodSegment::numPosDoF)
        .def("fullDoF",             &UmbrellaMesh::RodSegment::fullDoF)
        .def(py::pickle([](const UmbrellaMesh::RodSegment &s) { return py::make_tuple(s.startJoint, s.endJoint, s.rod, s.segmentType()); },
                        [](const py::tuple &t) {
                            if (t.size() != 4) throw std::runtime_error("Invalid UmbrellaMesh::RodSegment state!");
                            return UmbrellaMesh::RodSegment(t[0].cast<size_t>(), t[1].cast<size_t>(), t[2].cast<ElasticRod>(), t[3].cast<UmbrellaMesh::SegmentType>());
                        }))
        ;

    py::class_<UMTE>(m, "UmbrellaMeshTerminalEdgeSensitivity")
        .def(py::init<const UmbrellaMesh::Joint, size_t, const ElasticRod_T<Real> &, bool, bool>(), 
            py::arg("joint"),
            py::arg("si"),
            py::arg("rod"),
            py::arg("updatedSource"), 
            py::arg("evalHessian"))
        .def_readonly("jacobian", &UMTE::jacobian)
        .def_readonly("hessian", &UMTE::hessian)
        .def_readonly("delta_jacobian", &UMTE::delta_jacobian)
        .def_readonly("localSegmentIndex", &UMTE::localSegmentIndex)
        ;

    py::enum_<UmbrellaMesh::TerminalEdge>(m, "TerminalEdge")
        .value("Start", UmbrellaMesh::TerminalEdge::Start)
        .value("End",   UmbrellaMesh::TerminalEdge::End)
        ;

    umbrella_mesh
        .def(py::init<const UmbrellaMeshIO &, size_t>(), py::arg("umbrellamesh_io"), py::arg("subdivision") = 10)
        .def("energyStretch", &UmbrellaMesh::energyStretch, "Compute stretching energy")
        .def("energyBend",    &UmbrellaMesh::energyBend   , "Compute bending    energy")
        .def("energyTwist",   &UmbrellaMesh::energyTwist  , "Compute twisting   energy")
        .def("energyElastic",        py::overload_cast<UmbrellaMesh::EnergyType>(&UmbrellaMesh::energyElastic, py::const_), "Compute elastic energy", py::arg("energyType") = UmbrellaMesh::EnergyType::Full)
        .def("energyDeployment", &UmbrellaMesh::energyDeployment, "Compute deployment energy")
        .def("energyRepulsion", &UmbrellaMesh::energyRepulsion, "Compute repulsion energy")
        .def("energyAttraction", &UmbrellaMesh::energyAttraction, "Compute target attraction energy")
        .def("energy",        py::overload_cast<UmbrellaMesh::UmbrellaEnergyType, UmbrellaMesh::EnergyType>(&UmbrellaMesh::energy, py::const_), "Compute total energy", py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Full, py::arg("elasticEnergyType") = UmbrellaMesh::EnergyType::Full)
        .def("gradient", &UmbrellaMesh::gradient, "Elastic energy gradient", py::arg("updatedSource") = false, py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Full, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false)
        .def("hessianNNZ",             &UmbrellaMesh::hessianNNZ,             "Tight upper bound for nonzeros in the Hessian.",                           py::arg("variableDesignParameters") = false)
        .def("hessianSparsityPattern", &UmbrellaMesh::hessianSparsityPattern, "Compressed column matrix containing all potential nonzero Hessian entries", py::arg("variableDesignParameters") = false, py::arg("val") = 0.0)
        .def("hessian",  py::overload_cast<UmbrellaMesh::UmbrellaEnergyType, ElasticRod::EnergyType, bool>(&UmbrellaMesh::hessian, py::const_), "Elastic energy + deployment energy + repulsion energy hessian", py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Full, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false)
        .def("applyHessian", &UmbrellaMesh::applyHessian, "Umbrella energy Hessian-vector product formulas.", py::arg("v"), py::arg("variableDesignParameters") = false, py::arg("mask") = HessianComputationMask(), py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Full)

        // Linear Actuator
        .def("getLinearActuator", &UmbrellaMesh::getLinearActuator, py::return_value_policy::reference)
        .def("linearActuatorEnergy",   &UmbrellaMesh::linearActuatorEnergy)
        .def("linearActuatorGradient", &UmbrellaMesh::linearActuatorGradient)
        .def("linearActuatorHessian",  &UmbrellaMesh::linearActuatorHessian)
        .def("linearActuatorHessVec",  &UmbrellaMesh::linearActuatorHessVec, py::arg("v"))
        .def_property_readonly("plateHeights",  &UmbrellaMesh::plateHeights)
        .def("updateRotationParametrizations", &UmbrellaMesh::updateRotationParametrizations)
        
        .def("gradientPerArmRestlen", &UmbrellaMesh::gradientPerArmRestlen, "Elastic energy gradient for per segment rest length", py::arg("updatedSource") = false, py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Full, py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("hessianPerArmRestlen",  py::overload_cast<UmbrellaMesh::UmbrellaEnergyType, ElasticRod::EnergyType>(&UmbrellaMesh::hessianPerArmRestlen, py::const_), "Elastic energy  hessian for per segment rest length", py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Full, py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("applyHessianPerArmRestlen", &UmbrellaMesh::applyHessianPerArmRestlen, "Elastic energy Hessian-vector product formulas for per segment rest length.", py::arg("v"), py::arg("mask") = HessianComputationMask(), py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Full)

        // Deployment configuration
        .def_property("uniformDeploymentEnergyWeight", [](const UmbrellaMesh &um)             { return um.getUniformDeploymentEnergyWeight(); },
                                                [](      UmbrellaMesh &um, const Real weight) { um.setUniformDeploymentEnergyWeight(weight);  }, "Uniform weight for deployment energy")
        .def_property("deploymentEnergyWeight", [](const UmbrellaMesh &um)             { return um.getDeploymentEnergyWeight(); },
                                                [](      UmbrellaMesh &um, const Eigen::VectorXd weight) { um.setDeploymentEnergyWeight(weight);  }, "Weight for deployment energy")
        .def_property("targetDeploymentHeight", [](const UmbrellaMesh &um)                    { return um.getTargetDeploymentHeight(); },
                                                [](      UmbrellaMesh &um, const Real weight) { um.setTargetDeploymentHeight(weight);  }, "The target height of the umbrella. If the umbrella open to this height, the deployment energy for it will be zero.")
        .def_property("targetDeploymentHeightVector", [](const UmbrellaMesh &um)                    { return um.getTargetDeploymentHeightVector(); },
                                                [](      UmbrellaMesh &um, const Eigen::VectorXd heights) { um.setTargetDeploymentHeightVector(heights);  }, "The target height of the umbrella. If the umbrella open to this height, the deployment energy for it will be zero.")
        .def_property("deploymentForceType",    [](const UmbrellaMesh &um)                    { return um.getDeploymentForceType(); },
                                                [](      UmbrellaMesh &um, const UmbrellaMesh::DeploymentForceType dftype) { um.setDeploymentForceType(dftype);  }, "Choice of deployment force type between spring and constant")
        .def_property("repulsionEnergyWeight",  [](const UmbrellaMesh &um)                    { return um.getRepulsionEnergyWeight(); },
                                                [](      UmbrellaMesh &um, const Real weight) { um.setRepulsionEnergyWeight(weight);  }, "Weight for repulsion energy")
        .def_property("attractionWeight",       [](const UmbrellaMesh &um)                    { return um.getAttractionWeight(); },
                                                [](      UmbrellaMesh &um, const Real weight) { um.setAttractionWeight(weight);  }, "Weight for surface attraction energy")
        .def_property("angleBoundEnforcement",  &UmbrellaMesh::getAngleBoundEnforcement, &UmbrellaMesh::setAngleBoundEnforcement, "Method for enforcing the joint angle bounds")

        .def_property_readonly("umbrellaHeights", &UmbrellaMesh::getUmbrellaHeights)
        .def("updateSourceFrame", &UmbrellaMesh::updateSourceFrame, "Use the current reference frame as the source for parallel transport")
        .def("getDoFs",       &UmbrellaMesh::getDoFs)
        .def("setDoFs",       py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&UmbrellaMesh::setDoFs), py::arg("values"), py::arg("spatialCoherence") = false)
        .def("numDoF",           &UmbrellaMesh::numDoF)
        .def("numSegments",      &UmbrellaMesh::numSegments)
        .def("numArmSegments",   &UmbrellaMesh::numArmSegments)
        .def("numJoints",        &UmbrellaMesh::numJoints)
        .def("numRigidJoints",   &UmbrellaMesh::numRigidJoints)
        .def("numCenterlinePos", &UmbrellaMesh::numCenterlinePos)
        .def("numURH", &UmbrellaMesh::numURH)
        .def("numUmbrellas", &UmbrellaMesh::numUmbrellas)
        .def("numXJoints", &UmbrellaMesh::numXJoints)
        .def("segment", py::overload_cast<size_t>(&UmbrellaMesh::segment), py::return_value_policy::reference)
        .def("joint",   py::overload_cast<size_t>(&UmbrellaMesh::joint),   py::return_value_policy::reference)
        .def("segments", [](const UmbrellaMesh &um) { return py::make_iterator(um.segments().cbegin(), um.segments().cend()); })
        .def("joints",   [](const UmbrellaMesh &um) { return py::make_iterator(um.joints  ().cbegin(), um.joints  ().cend()); })
        .def("dofOffsetForJoint",          &UmbrellaMesh::dofOffsetForJoint,         py::arg("index"))
        .def("dofOffsetForSegment",        &UmbrellaMesh::dofOffsetForSegment,       py::arg("index"))
        .def("dofOffsetForCenterlinePos",  &UmbrellaMesh::dofOffsetForCenterlinePos, py::arg("index"))
        .def("restLenDofOffsetForSegment", &UmbrellaMesh::restLenDofOffsetForSegment, py::arg("index"))
        .def("designParameterDoFOffsetForJoint", &UmbrellaMesh::designParameterDoFOffsetForJoint, py::arg("index"))
        .def("setMaterial",   [](UmbrellaMesh &um, RodMaterial &mat) { um.setMaterial(mat); }, py::arg("material"))
        .def("setMaterial",   [](UmbrellaMesh &um, RodMaterial &am, RodMaterial &pm) { um.setMaterial(am, pm); }, py::arg("armMaterial"), py::arg("plateMaterial"))
        .def_property("averageJointAngle", [](const UmbrellaMesh &um)                   { return um.getAverageJointAngle(); },
                                           [](      UmbrellaMesh &um, const Real alpha) { um.setAverageJointAngle(alpha);   })
        .def("jointAngleDoFIndices",      &UmbrellaMesh::jointAngleDoFIndices)
        .def("rigidJointAngleDoFIndices", &UmbrellaMesh::rigidJointAngleDoFIndices)
        .def("centralJoint",              &UmbrellaMesh::centralJoint)
        .def("jointPositionDoFIndices",   &UmbrellaMesh::jointPositionDoFIndices)
        .def("getUmbrellaHeights",        &UmbrellaMesh::getUmbrellaHeights)
        .def("getArmUID",                 &UmbrellaMesh::getArmUID, py::arg("si"))
        .def("getArmIndexAt",             &UmbrellaMesh::getArmIndexAt, py::arg("si"))
        // Umbrella Rest Height Design Parameters

        // Design Optimization
        .def("getRestLengths",            &UmbrellaMesh::getRestLengths)
        .def("minRestLength",             &UmbrellaMesh::minRestLength)

        .def("getExtendedDoFs",     &UmbrellaMesh::getExtendedDoFs)
        .def("setExtendedDoFs",     &UmbrellaMesh::setExtendedDoFs, py::arg("values"), py::arg("spatialCoherence") = false)
        .def("numExtendedDoF",      &UmbrellaMesh::numExtendedDoF)
        .def("numExtendedDoFPARL",  &UmbrellaMesh::numExtendedDoFPARL)

        .def("getExtendedDoFsPARL", &UmbrellaMesh::getExtendedDoFsPARL)
        .def("setExtendedDoFsPARL", &UmbrellaMesh::setExtendedDoFsPARL, py::arg("values"), py::arg("spatialCoherence") = false)
        .def("getPerArmRestLength", &UmbrellaMesh::getPerArmRestLength)
        .def("setPerArmRestLength", &UmbrellaMesh::setPerArmRestLength, py::arg("values"))

        .def("designParameterSolveFixedVars", &UmbrellaMesh::designParameterSolveFixedVars)

        .def("set_design_parameter_config", &UmbrellaMesh::setDesignParameterConfig, py::arg("use_restLen"), py::arg("use_restKappa"), py::arg("update_designParams_cache") = true)
        .def("get_design_parameter_config", &UmbrellaMesh::getDesignParameterConfig)
        .def("getDesignParameters",         &UmbrellaMesh::getDesignParameters)
        .def("get_l0", &UmbrellaMesh::get_l0)
        .def("get_E0", &UmbrellaMesh::get_E0)

        // Target Surface Fitter
        .def("getTargetSurface",          &UmbrellaMesh::getTargetSurface)
        .def("getHoldClosestPointsFixed", &UmbrellaMesh::getHoldClosestPointsFixed)
        .def("setHoldClosestPointsFixed", &UmbrellaMesh::setHoldClosestPointsFixed, py::arg("setHoldClosestPointsFixed"))
        .def("scaleInputPosWeights",      &UmbrellaMesh::scaleInputPosWeights, py::arg("inputPosWeight"), py::arg("bdryMultiplier") = 1, py::arg("featureMultiplier") = 1, py::arg("additional_feature_pts") = std::vector<size_t>())
        .def("XJointPositions",           &UmbrellaMesh::XJointPositions)
        .def("XJointTgtPositions",        &UmbrellaMesh::XJointTgtPositions)
        .def("IsQueryPtBoundary",        &UmbrellaMesh::IsQueryPtBoundary)

        // Vibration Mode Analysis
        .def("massMatrix", py::overload_cast<bool, bool>(&UmbrellaMesh::massMatrix, py::const_), py::arg("updatedSource") = false, py::arg("useLumped") = false)
        .def("characteristicLength", &UmbrellaMesh::characteristicLength)
        .def("approxLinfVelocity",   &UmbrellaMesh::approxLinfVelocity)

        // Deployment 
        .def("getUmbrellaCenterJi", &UmbrellaMesh::getUmbrellaCenterJi, py::arg("ui"), py::arg("ti"))
        // Visualization
        .def("saveVisualizationGeometry",   &UmbrellaMesh::saveVisualizationGeometry, py::arg("path"), py::arg("averagedMaterialFrames") = false)
        .def("visualizationGeometry", &getVisualizationGeometry<UmbrellaMesh>, py::arg("averagedMaterialFrames") = true, py::arg("averagedCrossSections") = true)
        .def("sqrtBendingEnergies", &UmbrellaMesh::sqrtBendingEnergies)
        .def("stretchingStresses",  &UmbrellaMesh:: stretchingStresses)
        .def("maxBendingStresses",  &UmbrellaMesh:: maxBendingStresses)
        .def("minBendingStresses",  &UmbrellaMesh:: minBendingStresses)
        .def("twistingStresses",    &UmbrellaMesh::   twistingStresses)
        .def("maxVonMisesStresses",  &UmbrellaMesh:: maxVonMisesStresses)
        .def("visualizationField", [](const UmbrellaMesh &r, const std::vector<Eigen::VectorXd>  &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
        .def("visualizationField", [](const UmbrellaMesh &r, const std::vector<Eigen::MatrixX3d> &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))

        // Stress analysis
        .def("surfaceStressLpNorm",     &UmbrellaMesh::                  surfaceStressLpNorm, py::arg("stressType"),                           py::arg("p"), py::arg("takeRoot") = true)
        .def("gradSurfaceStressLpNorm", &UmbrellaMesh::gradSurfaceStressLpNormPerEdgeRestLen, py::arg("stressType"), py::arg("updatedSource"), py::arg("p"), py::arg("takeRoot") = true)

        // Forces
        .def("rivetForces", &UmbrellaMesh::rivetForces, "Compute the forces exerted on each joint", py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Elastic, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("needTorque") = true)
        .def("UmbrellaRivetNetForceAndTorques", &UmbrellaMesh::UmbrellaRivetNetForceAndTorques, "Compute the forces exerted on the centers of the umbrella plates.", py::arg("umbrellaEnergyType") = UmbrellaMesh::UmbrellaEnergyType::Elastic, py::arg("energyType") = ElasticRod::EnergyType::Full)

        // Debug
        .def("getTerminalEdgeSensitivity", py::overload_cast<size_t, UmbrellaMesh::TerminalEdge, bool, bool>(&UmbrellaMesh::getTerminalEdgeSensitivity), py::arg("si"), py::arg("which"), py::arg("updatedSource"), py::arg("evalHessian"))
        .def("getTerminalEdgeSensitivity", py::overload_cast<size_t, UmbrellaMesh::TerminalEdge, bool, const Eigen::VectorXd &>(&UmbrellaMesh::getTerminalEdgeSensitivity), py::arg("si"), py::arg("which"), py::arg("updatedSource"), py::arg("delta_params"))
        .def_readwrite("disableRotationParametrizationUpdates", &UmbrellaMesh::disableRotationParametrizationUpdates)
        .def("rigidJoints", &UmbrellaMesh::rigidJoints);
        // Pickle
        
        addSerializationBindings<UmbrellaMesh, PyUM, UmbrellaMesh::StateV1>(umbrella_mesh);
        ;
    ////////////////////////////////////////////////////////////////////////////////
    // Equilibrium solver
    ////////////////////////////////////////////////////////////////////////////////
    m.attr("TARGET_ANGLE_NONE") = py::float_(TARGET_ANGLE_NONE);

    m.def("compute_equilibrium",
        [](UmbrellaMesh &umbrella_mesh, Real targetAverageAngle, const
            Eigen::VectorXd &externalForces, const NewtonOptimizerOptions
            &options, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb,
            Real elasticEnergyIncreaseFactorLimit) {
            py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
            py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
            auto cb = callbackWrapper(pcb);
            return compute_equilibrium(umbrella_mesh, targetAverageAngle, externalForces, options, fixedVars, cb, elasticEnergyIncreaseFactorLimit);
        },
        py::arg("umbrella_mesh"),
        py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
        py::arg("externalForces") = std::vector<size_t>(),
        py::arg("options") = NewtonOptimizerOptions(),
        py::arg("fixedVars") = std::vector<size_t>(),
        py::arg("callback") = nullptr,
        py::arg("elasticEnergyIncreaseFactorLimit") = 2.0
    );

    m.def("get_equilibrium_optimizer",
          [](UmbrellaMesh &umbrella_mesh, Real targetAverageAngle, const std::vector<size_t> &fixedVars) { return get_equilibrium_optimizer(umbrella_mesh, targetAverageAngle, fixedVars); },
          py::arg("umbrella_mesh"),
          py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
          py::arg("fixedVars") = std::vector<size_t>()
    );

    ////////////////////////////////////////////////////////////////////////////////
    // Design Parameter Solve
    ////////////////////////////////////////////////////////////////////////////////
    bindDesignParameterProblem<UmbrellaMesh>(detail_module, "UmbrellaMesh");
    m.def("DesignParameterProblem", [](UmbrellaMesh &umbrella) {
        return py::cast(new DesignParameterProblem<UmbrellaMesh>(umbrella), py::return_value_policy::take_ownership);
    }, py::arg("umbrella"));

    m.def("designParameter_solve",
          [](UmbrellaMesh &umbrella, const NewtonOptimizerOptions &opts, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb, Real E0) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              return designParameter_solve(umbrella, opts, fixedVars, cb, E0);
          },
          py::arg("umbrella"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr,
          py::arg("E0") = -1
    );
    m.def("get_designParameter_optimizer", [](UmbrellaMesh &um, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb, Real E0) {
            return get_designParameter_optimizer(um, fixedVars, callbackWrapper(pcb), E0);
         },
          py::arg("umbrella"),
          py::arg("fixedVars") = std::vector<size_t>(), py::arg("callback") = nullptr,
          py::arg("E0") = -1
    );
    m.def("designParameter_problem",
          [](UmbrellaMesh &umbrella, const std::vector<size_t> &fixedVars, Real E0) -> std::unique_ptr<NewtonProblem> {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return designParameter_problem(umbrella, fixedVars, E0);
          },
          py::arg("umbrella"),
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("E0") = -1
    );

    ////////////////////////////////////////////////////////////////////////////////
    // UmbrellaMesh IO
    ////////////////////////////////////////////////////////////////////////////////
    using V3d = UmbrellaMeshIO::V3d;
    auto pyUMIO = py::class_<UmbrellaMeshIO>(m, "UmbrellaMeshIO");
    py::class_<UmbrellaMeshIO::Joint>(pyUMIO, "Joint")
        .def(py::init<UmbrellaMeshIO::JointType,
                      const V3d &, const V3d &, const V3d &,
                      double, const std::vector<size_t> &, const V3d &>(),
             py::arg("type"), py::arg("position"), py::arg("bisector"), py::arg("normal"),
             py::arg("alpha"), py::arg("umbrella_ID"), py::arg("correspondence"))
        .def_readwrite("type",        &UmbrellaMeshIO::Joint::type)
        .def_readwrite("position",    &UmbrellaMeshIO::Joint::position)
        .def_readwrite("bisector",    &UmbrellaMeshIO::Joint::bisector)
        .def_readwrite("normal",      &UmbrellaMeshIO::Joint::normal)
        .def_readwrite("alpha",       &UmbrellaMeshIO::Joint::alpha)
        .def_readwrite("umbrella_ID", &UmbrellaMeshIO::Joint::umbrella_ID)
        .def_readwrite("tgt_pos", &UmbrellaMeshIO::Joint::tgt_pos)
        ;
    py::class_<UmbrellaMeshIO::JointConnection>(pyUMIO, "JointConnection")
        .def(py::init<size_t, bool, const V3d &>(),
                py::arg("joint_index"), py::arg("is_A"), py::arg("midpoint_offset"))
        .def_readwrite("joint_index",     &UmbrellaMeshIO::JointConnection::joint_index)
        .def_readwrite("is_A",            &UmbrellaMeshIO::JointConnection::is_A)
        .def_readwrite("midpoint_offset", &UmbrellaMeshIO::JointConnection::midpoint_offset)
        ;
    py::class_<UmbrellaMeshIO::Segment>(pyUMIO, "Segment")
        .def(py::init<UmbrellaMeshIO::SegmentType,
                      const std::vector<UmbrellaMeshIO::JointConnection> &,
                      const V3d &>(),
             py::arg("type"), py::arg("endpoint"),
             py::arg("normal"))
        .def_readwrite("type",     &UmbrellaMeshIO::Segment::type)
        .def_readwrite("endpoint", &UmbrellaMeshIO::Segment::endpoint)
        .def_readwrite("normal",   &UmbrellaMeshIO::Segment::normal)
        ;
    py::class_<UmbrellaMeshIO::Umbrella>(pyUMIO, "Umbrella")
        .def(py::init<size_t, size_t, const V3d &>(),
             py::arg("top_joint"), py::arg("bottom_joint"), py::arg("correspondence"))
        .def_readwrite("top_joint",    &UmbrellaMeshIO::Umbrella::top_joint)
        .def_readwrite("bottom_joint", &UmbrellaMeshIO::Umbrella::bottom_joint)
        .def_readwrite("tgt_pos", &UmbrellaMeshIO::Umbrella::tgt_pos)
        ;
    pyUMIO
        .def(py::init<const std::vector<UmbrellaMeshIO::Joint> &,
                      const std::vector<UmbrellaMeshIO::Segment> &,
                      const std::vector<UmbrellaMeshIO::Umbrella> &,
                      const std::vector<std::vector<size_t> > &,
                      const std::vector<double> &,
                      const Eigen::MatrixXd &,
                      const Eigen::MatrixXi &>(),
            py::arg("joints"), py::arg("segments"), py::arg("umbrellas"), py::arg("umbrella_connectivity"), py::arg("material_params"), py::arg("target_v"), py::arg("target_f"))
        .def("validate",            &UmbrellaMeshIO::validate)
        .def_readwrite("joints",    &UmbrellaMeshIO::joints)
        .def_readwrite("segments",  &UmbrellaMeshIO::segments)
        .def_readwrite("umbrellas", &UmbrellaMeshIO::umbrellas)
        .def_readwrite("umbrella_connectivity", &UmbrellaMeshIO::umbrella_connectivity)
        .def_readwrite("material_params", &UmbrellaMeshIO::material_params)
        .def_readwrite("target_v", &UmbrellaMeshIO::target_v)
        .def_readwrite("target_f", &UmbrellaMeshIO::target_f)
        ;
}

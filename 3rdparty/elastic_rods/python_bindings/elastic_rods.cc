#include <iostream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <memory>
#include "../ElasticRod.hh"
#include "../compute_equilibrium.hh"

#include "../CrossSection.hh"
#include "../cross_sections/Custom.hh"
#include "../CrossSectionMesh.hh"
#include "../RectangularBox.hh"
#include "visualization.hh"


#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

template <typename T>
std::string to_string_with_precision(const T &val, const int n = 6) {
    std::ostringstream ss;
    ss << std::setprecision(n) << val;
    return ss.str();

}
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

// Hack around a limitation of pybind11 where we cannot specify argument passing policies and
// pybind11 tries to make a copy if the passed instance is not already registered:
//      https://github.com/pybind/pybind11/issues/1200
// We therefore make our Python callback interface use a raw pointer to forbid this copy (which
// causes an error since NewtonProblem is not copyable).
using PyCallbackFunction = std::function<void(NewtonProblem *, size_t)>;
CallbackFunction callbackWrapper(const PyCallbackFunction &pcb) {
    return [pcb](NewtonProblem &p, size_t i) -> void { if (pcb) pcb(&p, i); };
}


PYBIND11_MODULE(elastic_rods, m) {
    m.doc() = "Elastic Rods Codebase";

    py::module::import("MeshFEM");
    py::module::import("mesh");
    py::module::import("sparse_matrices");
    py::module::import("py_newton_optimizer");
    
    py::module detail_module = m.def_submodule("detail");

    ////////////////////////////////////////////////////////////////////////////////
    // ElasticRods and nested classes
    ////////////////////////////////////////////////////////////////////////////////
    auto elastic_rod = py::class_<ElasticRod>(m, "ElasticRod");

    py::enum_<ElasticRod::EnergyType>(m, "EnergyType")
        .value("Full",    ElasticRod::EnergyType::Full   )
        .value("Bend",    ElasticRod::EnergyType::Bend   )
        .value("Twist",   ElasticRod::EnergyType::Twist  )
        .value("Stretch", ElasticRod::EnergyType::Stretch)
        ;

    py::enum_<ElasticRod::BendingEnergyType>(m, "BendingEnergyType")
        .value("Bergou2010", ElasticRod::BendingEnergyType::Bergou2010)
        .value("Bergou2008", ElasticRod::BendingEnergyType::Bergou2008)
        ;


    py::class_<GradientStencilMaskCustom>(m, "GradientStencilMaskCustom")
        .def(py::init<>())
        .def_readwrite("edgeStencilMask", &GradientStencilMaskCustom::edgeStencilMask)
        .def_readwrite("vtxStencilMask",  &GradientStencilMaskCustom::vtxStencilMask)
        ;

    py::class_<HessianComputationMask>(m, "HessianComputationMask")
        .def(py::init<>())
        .def_readwrite("dof_in",              &HessianComputationMask::dof_in)
        .def_readwrite("dof_out",             &HessianComputationMask::dof_out)
        .def_readwrite("designParameter_in",  &HessianComputationMask::designParameter_in)
        .def_readwrite("designParameter_out", &HessianComputationMask::designParameter_out)
        .def_readwrite("skipBRods",           &HessianComputationMask::skipBRods)
        ;

    py::class_<DesignParameterConfig>(m, "DesignParameterConfig")
        .def(py::init<>())
        .def_readonly("restLen", &DesignParameterConfig::restLen)
        .def_readonly("restKappa", &DesignParameterConfig::restKappa)
        .def(py::pickle([](const DesignParameterConfig &dpc) { return py::make_tuple(dpc.restLen, dpc.restKappa); },
                        [](const py::tuple &t) {
                        if (t.size() != 2) throw std::runtime_error("Invalid DesignParameterConfig!");
                            DesignParameterConfig dpc; 
                            dpc.restLen = t[0].cast<bool>();
                            dpc.restKappa = t[1].cast<bool>();
                            return dpc;
                        }));

    elastic_rod
        .def(py::init<std::vector<Point3D>>())
        .def("__repr__", [](const ElasticRod &e) { return "Elastic rod with " + std::to_string(e.numVertices()) + " points and " + std::to_string(e.numEdges()) + " edges"; })
        .def("setDeformedConfiguration", py::overload_cast<const std::vector<Point3D> &, const std::vector<Real> &>(&ElasticRod::setDeformedConfiguration))
        .def("setDeformedConfiguration", py::overload_cast<const ElasticRod::DeformedState &>(&ElasticRod::setDeformedConfiguration))
        .def("deformedPoints", &ElasticRod::deformedPoints)
        .def("restDirectors",  &ElasticRod::restDirectors)
        .def("thetas",         &ElasticRod::thetas)
        .def("setMaterial",    py::overload_cast<const             RodMaterial  &>(&ElasticRod::setMaterial))
        .def("setMaterial",    py::overload_cast<const std::vector<RodMaterial> &>(&ElasticRod::setMaterial))
        .def("setLinearlyInterpolatedMaterial", &ElasticRod::setLinearlyInterpolatedMaterial, py::arg("startMat"), py::arg("endMat"))
        .def("material", py::overload_cast<size_t>(&ElasticRod::material, py::const_), py::return_value_policy::reference)
        .def("set_design_parameter_config", &ElasticRod::setDesignParameterConfig, py::arg("use_restLen"), py::arg("use_restKappa"))
        .def("get_design_parameter_config", &ElasticRod::getDesignParameterConfig)
        .def("setRestLengths", &ElasticRod::setRestLengths, py::arg("val"))
        .def("setRestKappas", &ElasticRod::setRestKappas, py::arg("val"))
        .def("numRestKappaVars", &ElasticRod::numRestKappaVars)
        .def("setRestKappaVars", &ElasticRod::setRestKappaVars, py::arg("params"))
        .def("getRestKappaVars", &ElasticRod::getRestKappaVars)
        .def("restKappas", py::overload_cast<>(&ElasticRod::restKappas, py::const_))
        .def("restPoints", &ElasticRod::restPoints)

        // Outputs mesh with normals
        .def("visualizationGeometry",             &getVisualizationGeometry<ElasticRod>, py::arg("averagedMaterialFrames") = true, py::arg("averagedCrossSections") = true)
        .def("visualizationGeometryHeightColors", &getVisualizationGeometryCSHeightField<ElasticRod>, "Get a per-visualization-vertex field representing height above the centerline")

        .def("rawVisualizationGeometry", [](ElasticRod &r, const bool averagedMaterialFrames, const bool averagedCrossSections) {
                std::vector<MeshIO::IOVertex > vertices;
                std::vector<MeshIO::IOElement> quads;
                r.visualizationGeometry(vertices, quads, averagedMaterialFrames, averagedCrossSections);
                const size_t nv = vertices.size(),
                             ne = quads.size();
                Eigen::MatrixX3d V(nv, 3);
                Eigen::MatrixX4i F(ne, 4);

                for (size_t i = 0; i < nv; ++i) V.row(i) = vertices[i].point;
                for (size_t i = 0; i < ne; ++i) {
                    const auto &q = quads[i];
                    if (q.size() != 4) throw std::runtime_error("Expected quads");
                    F.row(i) << q[0], q[1], q[2], q[3];
                }

                return std::make_pair(V, F);
            }, py::arg("averagedMaterialFrames") = false, py::arg("averagedCrossSections") = true)
        .def("saveVisualizationGeometry", &ElasticRod::saveVisualizationGeometry, py::arg("path"), py::arg("averagedMaterialFrames") = false, py::arg("averagedCrossSections") = false)
        .def("writeDebugData", &ElasticRod::writeDebugData)

        .def("deformedConfiguration", py::overload_cast<>(&ElasticRod::deformedConfiguration, py::const_), py::return_value_policy::reference)
        .def("updateSourceFrame", &ElasticRod::updateSourceFrame)

        .def("numEdges",    &ElasticRod::numEdges)
        .def("numVertices", &ElasticRod::numVertices)

        .def("numDoF",  &ElasticRod::numDoF)
        .def("getDoFs", &ElasticRod::getDoFs)
        .def("setDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&ElasticRod::setDoFs), py::arg("values"))

        .def("posOffset",     &ElasticRod::posOffset)
        .def("thetaOffset",   &ElasticRod::thetaOffset)
        // TODO (Samara)
        .def("restLenOffset", &ElasticRod::restLenOffset)
        .def("designParameterOffset", &ElasticRod::designParameterOffset)

        .def("numExtendedDoF",  &ElasticRod::numExtendedDoF)
        .def("getExtendedDoFs", &ElasticRod::getExtendedDoFs)
        .def("setExtendedDoFs", &ElasticRod::setExtendedDoFs)
        .def("lengthVars"     , &ElasticRod::lengthVars, py::arg("variableRestLen") = false)

        .def("totalRestLength",           &ElasticRod::totalRestLength)
        .def("restLengths",               &ElasticRod::restLengths)
        // Determine the deformed position at curve parameter 0.5
        .def_property_readonly("midpointPosition", [](const ElasticRod &e) -> Point3D {
                size_t ne = e.numEdges();
                // Midpoint is in the middle of an edge for odd numbers of edges,
                // at a vertex for even numbers of edges.
                if (ne % 2) return 0.5 * (e.deformedPoint(ne / 2) + e.deformedPoint(ne / 2 + 1));
                else        return e.deformedPoint(ne / 2);
            })
        // Determine the deformed material frame vector d2 at curve parameter 0.5
        .def_property_readonly("midpointD2", [](const ElasticRod &e) -> Vector3D {
                size_t ne = e.numEdges();
                // Midpoint is in the middle of an edge for odd numbers of edges,
                // at a vertex for even numbers of edges.
                if (ne % 2) return e.deformedMaterialFrameD2(ne / 2);
                else        return 0.5 * (e.deformedMaterialFrameD2(ne / 2 - 1) + e.deformedMaterialFrameD2(ne / 2));
            })

        .def_property("bendingEnergyType", [](const ElasticRod &e) { return e.bendingEnergyType(); },
                                           [](ElasticRod &e, ElasticRod::BendingEnergyType type) { e.setBendingEnergyType(type); })
        .def("energyStretch", &ElasticRod::energyStretch, "Compute stretching energy")
        .def("energyBend",    &ElasticRod::energyBend   , "Compute bending    energy")
        .def("energyTwist",   &ElasticRod::energyTwist  , "Compute twisting   energy")
        .def("energy",        py::overload_cast<ElasticRod::EnergyType>(&ElasticRod::energy, py::const_), "Compute elastic energy", py::arg("energyType") = ElasticRod::EnergyType::Full)

        .def("gradEnergyStretch", &ElasticRod::gradEnergyStretch<GradientStencilMaskCustom>, "Compute stretching energy gradient"                                                                                        , py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradEnergyBend",    &ElasticRod::gradEnergyBend   <GradientStencilMaskCustom>, "Compute bending    energy gradient", py::arg("updatedSource") = false                                                      , py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradEnergyTwist",   &ElasticRod::gradEnergyTwist  <GradientStencilMaskCustom>, "Compute twisting   energy gradient", py::arg("updatedSource") = false                                                      , py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradient",          &ElasticRod::gradient         <GradientStencilMaskCustom>, "Compute elastic    energy gradient", py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())

        .def("hessianNNZ",             &ElasticRod::hessianNNZ,             "Tight upper bound for nonzeros in the Hessian.", py::arg("variableDesignParameters") = false)
        .def("hessianSparsityPattern", &ElasticRod::hessianSparsityPattern, "Compressed column matrix containing all potential nonzero Hessian entries", py::arg("variableDesignParameters") = false, py::arg("val") = 0.0)

        .def("hessian",           [](const ElasticRod &e, ElasticRod::EnergyType eType, bool variableDesignParameters) { return e.hessian(eType, variableDesignParameters); }, "Compute elastic energy Hessian", py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false)

        .def("applyHessian", &ElasticRod::applyHessian, "Elastic energy Hessian-vector product formulas.", py::arg("v"), py::arg("variableDesignParameters") = false, py::arg("mask") = HessianComputationMask())

        .def("massMatrix",        py::overload_cast<>(&ElasticRod::massMatrix, py::const_))
        .def("lumpedMassMatrix",  &ElasticRod::lumpedMassMatrix)

        .def("characteristicLength", &ElasticRod::characteristicLength)
        .def("approxLinfVelocity",   &ElasticRod::approxLinfVelocity)

        .def("bendingStiffnesses",  py::overload_cast<>(&ElasticRod::bendingStiffnesses,  py::const_), py::return_value_policy::reference)
        .def("twistingStiffnesses", py::overload_cast<>(&ElasticRod::twistingStiffnesses, py::const_), py::return_value_policy::reference)

        .def("stretchingStresses",      &ElasticRod::     stretchingStresses)
        .def("bendingStresses",         &ElasticRod::        bendingStresses)
        .def("minBendingStresses",      &ElasticRod::     minBendingStresses)
        .def("maxBendingStresses",      &ElasticRod::     maxBendingStresses)
        .def("twistingStresses",        &ElasticRod::       twistingStresses)
        .def("maxStresses",             &ElasticRod::            maxStresses, py::arg("stressType"))
        .def("surfaceStressLpNorm",     &ElasticRod::    surfaceStressLpNorm, py::arg("stressType"), py::arg("p"),                               py::arg("takeRoot") = true)
        .def("gradSurfaceStressLpNorm", &ElasticRod::gradSurfaceStressLpNorm, py::arg("stressType"), py::arg("p"), py::arg("updateSourceFrame"), py::arg("takeRoot") = true)

        .def("edgeMaterials",    &ElasticRod::edgeMaterials)

        .def("visualizationField", [](const ElasticRod &r, const Eigen::VectorXd  &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
        .def("visualizationField", [](const ElasticRod &r, const Eigen::MatrixX3d &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))

        .def(py::pickle([](const ElasticRod &r) { return py::make_tuple(r.restPoints(), r.restDirectors(), r.restKappas(), r.restTwists(), r.restLengths(),
                                                         r.edgeMaterials(),
                                                         r.bendingStiffnesses(),
                                                         r.twistingStiffnesses(),
                                                         r.stretchingStiffnesses(),
                                                         r.bendingEnergyType(),
                                                         r.deformedConfiguration(),
                                                         r.densities(),
                                                         r.initialMinRestLength()); },
                        [](const py::tuple &t) {
                        if ((t.size() < 11) || (t.size() > 13)) throw std::runtime_error("Invalid state!");
                            ElasticRod r              (t[ 0].cast<std::vector<Point3D>              >());
                            r.setRestDirectors        (t[ 1].cast<std::vector<ElasticRod::Directors>>());
                            r.setRestKappas           (t[ 2].cast<ElasticRod::StdVectorVector2D     >());
                            r.setRestTwists           (t[ 3].cast<std::vector<Real>                 >());
                            r.setRestLengths          (t[ 4].cast<std::vector<Real>                 >());

                            // Support old pickling format where only a RodMaterial was written instead of a vector of rod materials.
                            try         { r.setMaterial(t[ 5].cast<std::vector<RodMaterial>>()); }
                            catch (...) { r.setMaterial(t[ 5].cast<            RodMaterial >()); }

                            r.setBendingStiffnesses   (t[ 6].cast<std::vector<RodMaterial::BendingStiffness>>());
                            r.setTwistingStiffnesses  (t[ 7].cast<std::vector<Real>                         >());
                            r.setStretchingStiffnesses(t[ 8].cast<std::vector<Real>                         >());
                            r.setBendingEnergyType    (t[ 9].cast<ElasticRod::BendingEnergyType             >());
                            r.setDeformedConfiguration(t[10].cast<ElasticRod::DeformedState                 >());

                            // Support old pickling format where densities were absent.
                            if (t.size() > 11)
                                r.setDensities(t[11].cast<std::vector<Real>>());

                            // Support old pickling format where densities were absent.
                            if (t.size() > 12)
                                r.setInitialMinRestLen(t[12].cast<Real>());

                            return r;
                        }))
        ;

    // Note: the following bindings do not get used because PyBind thinks ElasticRod::Gradient is
    // just an Eigen::VectorXd. Also, they produce errors on Intel compilers.
    // py::class_<ElasticRod::Gradient>(elastic_rod, "Gradient")
    //     .def("__repr__", [](const ElasticRod::Gradient &g) { return "Elastic rod gradient with l2 norm " + to_string_with_precision(g.norm()); })
    //     .def_property_readonly("values", [](const ElasticRod::Gradient &g) { return Eigen::VectorXd(g); })
    //     .def("gradPos",   [](const ElasticRod::Gradient &g, size_t i) { return g.gradPos(i); })
    //     .def("gradTheta", [](const ElasticRod::Gradient &g, size_t j) { return g.gradTheta(j); })
    //     ;

    py::class_<ElasticRod::DeformedState>(elastic_rod, "DeformedState")
        .def("__repr__", [](const ElasticRod::DeformedState &) { return "Deformed state of an elastic rod (ElasticRod::DeformedState)."; })
        .def_readwrite("referenceDirectors", &ElasticRod::DeformedState::referenceDirectors)
        .def_readwrite("referenceTwist",     &ElasticRod::DeformedState::referenceTwist)
        .def_readwrite("tangent",            &ElasticRod::DeformedState::tangent)
        .def_readwrite("materialFrame",      &ElasticRod::DeformedState::materialFrame)
        .def_readwrite("kb",                 &ElasticRod::DeformedState::kb)
        .def_readwrite("kappa",              &ElasticRod::DeformedState::kappa)
        .def_readwrite("per_corner_kappa",   &ElasticRod::DeformedState::per_corner_kappa)
        .def_readwrite("len",                &ElasticRod::DeformedState::len)

        .def_readwrite("sourceTangent"           , &ElasticRod::DeformedState::sourceTangent)
        .def_readwrite("sourceReferenceDirectors", &ElasticRod::DeformedState::sourceReferenceDirectors)
        .def_readwrite("sourceMaterialFrame"     , &ElasticRod::DeformedState::sourceMaterialFrame)
        .def_readwrite("sourceReferenceTwist"    , &ElasticRod::DeformedState::sourceReferenceTwist)

        .def("updateSourceFrame", &ElasticRod::DeformedState::updateSourceFrame)

        .def("setReferenceTwist", &ElasticRod::DeformedState::setReferenceTwist)

        .def(py::pickle([](const ElasticRod::DeformedState &dc) { return py::make_tuple(dc.points(), dc.thetas(), dc.sourceTangent, dc.sourceReferenceDirectors, dc.sourceTheta, dc.sourceReferenceTwist); },
                        [](const py::tuple &t) {
                        // sourceReferenceTwist is optional for backwards compatibility
                        if (t.size() != 5 && t.size() != 6) throw std::runtime_error("Invalid state!");
                            ElasticRod::DeformedState dc;
                            const auto &pts             = t[0].cast<std::vector<Point3D              >>();
                            const auto &thetas          = t[1].cast<std::vector<Real                 >>();
                            dc.sourceTangent            = t[2].cast<std::vector<Vector3D             >>();
                            dc.sourceReferenceDirectors = t[3].cast<std::vector<ElasticRod::Directors>>();
                            dc.sourceTheta              = t[4].cast<std::vector<Real                 >>();
                            if (t.size() > 5)
                                dc.sourceReferenceTwist = t[5].cast<std::vector<Real                 >>();
                            else dc.sourceReferenceTwist.assign(thetas.size(), 0);

                            dc.update(pts, thetas);
                            return dc;
                        }))
        ;

    py::class_<ElasticRod::Directors>(elastic_rod, "Directors")
        .def("__repr__", [](const ElasticRod::Directors &dirs) { return "{ d1: [" + to_string_with_precision(dirs.d1.transpose()) + "], d2: [" + to_string_with_precision(dirs.d2.transpose()) + "] }"; })
        .def_readwrite("d1", &ElasticRod::Directors::d1)
        .def_readwrite("d2", &ElasticRod::Directors::d2)
        .def(py::pickle([](const ElasticRod::Directors &d) { return py::make_tuple(d.d1, d.d2); },
                        [](const py::tuple &t) {
                        if (t.size() != 2) throw std::runtime_error("Invalid state!");
                        return ElasticRod::Directors(
                                t[0].cast<Vector3D>(),
                                t[1].cast<Vector3D>());
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // CrossSection
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<CrossSection>(m, "CrossSection")
        .def_static("construct", &CrossSection::construct, py::arg("type"), py::arg("E"), py::arg("nu"), py::arg("params"))
        .def_static("fromContour", [](const std::string &path, Real E, Real nu, Real scale) -> std::unique_ptr<CrossSection> {
                auto result = std::make_unique<CrossSections::Custom>(path, scale);
                result->E = E, result->nu = nu;
                return result;
            }, py::arg("path"), py::arg("E"), py::arg("nu"), py::arg("scale") = 1.0)

        .def("boundary",  &CrossSection::boundary)
        .def("interior",  [](const CrossSection &cs, Real triArea) {
                std::vector<MeshIO::IOVertex > vertices;
                std::vector<MeshIO::IOElement> elements;
                std::tie(vertices, elements) = cs.interior(triArea);
                return std::make_shared<CrossSectionMesh::Base>(elements, vertices); // must match the container type of MeshFEM's bindings or we silently get a memory bug!
            }, py::arg("triArea") = 0.001)

        .def("numParams", &CrossSection::numParams)
        .def("setParams", &CrossSection::setParams, py::arg("p"))
        .def("params",    &CrossSection::params)

        .def("holePts",   &CrossSection::holePts)

        .def("copy", &CrossSection::copy)
        .def_static("lerp", &CrossSection::lerp, py::arg("cs_a"), py::arg("cs_b"), py::arg("alpha"))

        .def_readwrite("E", &CrossSection::E)
        .def_readwrite("nu", &CrossSection::nu)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // CrossSectionStressAnalysis
    ////////////////////////////////////////////////////////////////////////////////
    using CSA = CrossSectionStressAnalysis;
    py::class_<CSA, std::shared_ptr<CSA>> pyCSA(m, "CrossSectionStressAnalysis");
    py::enum_<CSA::StressType>(pyCSA, "StressType")
        .value("VonMises",     CSA::StressType::VonMises)
        .value("MaxMag",       CSA::StressType::MaxMag)
        .value("MaxPrincipal", CSA::StressType::MaxPrincipal)
        .value("MinPrincipal", CSA::StressType::MinPrincipal)
        .value("ZStress",      CSA::StressType::ZStress)
        ;

    pyCSA
        .def("maxStress", &CSA::maxStress<double>, py::arg("type"), py::arg("tau"), py::arg("curvatureNormal"), py::arg("stretching_strain"))
        .def_readonly("boundaryV",            &CSA::boundaryV)
        .def_readonly("boundaryE",            &CSA::boundaryE)
        .def_readonly("unitTwistShearStrain", &CSA::unitTwistShearStrain)
        .def_static("stressMeasure", [](CSA::StressType type, const Eigen::Vector2d &shearStress, Real sigma_zz, bool squared) {
                    if (squared) return CSA::stressMeasure< true>(type, shearStress, sigma_zz);
                    else         return CSA::stressMeasure<false>(type, shearStress, sigma_zz);
                }, py::arg("type"), py::arg("shearStress"), py::arg("sigma_zz"), py::arg("squared"))
        .def_static("gradStressMeasure", [](CSA::StressType type, const Eigen::Vector2d &shearStress, Real sigma_zz, bool squared) {
                    Eigen::Vector2d grad_shearStress;
                    Real grad_sigma_zz;
                    if (squared) { CSA::gradStressMeasure< true>(type, shearStress, sigma_zz, grad_shearStress, grad_sigma_zz); return std::make_pair(grad_shearStress, grad_sigma_zz); }
                    else         { CSA::gradStressMeasure<false>(type, shearStress, sigma_zz, grad_shearStress, grad_sigma_zz); return std::make_pair(grad_shearStress, grad_sigma_zz); }
                }, py::arg("type"), py::arg("shearStress"), py::arg("sigma_zz"), py::arg("squared"))
        .def(py::pickle([](const CSA &csa) {
                return std::make_tuple(csa.boundaryV, csa.boundaryE,
                                       csa.unitTwistShearStrain, csa.youngModulus,
                                       csa.shearModulus);
            },
            [](const std::tuple<CrossSection::AlignedPointCollection, CrossSection::EdgeCollection,
                                Eigen::MatrixX2d, Real, Real> &t) {
                return std::make_shared<CSA>(std::get<0>(t), std::get<1>(t),
                                             std::get<2>(t), std::get<3>(t),
                                             std::get<4>(t));
            }))
        ;
    // Stress visualization binding must come after StressType is bound...
    elastic_rod.def("stressVisualization", [](const ElasticRod &e, bool amf, bool acs, CrossSectionStressAnalysis::StressType t) { return getVisualizationWithStress(e, amf, acs, t); }, py::arg("averagedMaterialFrames") = true, py::arg("averagedCrossSections") = true, py::arg("stressType") = CrossSectionStressAnalysis::StressType::VonMises)
        ;


    ////////////////////////////////////////////////////////////////////////////////
    // RodMaterial
    ////////////////////////////////////////////////////////////////////////////////
    py::enum_<RodMaterial::StiffAxis>(m, "StiffAxis")
        .value("D1", RodMaterial::StiffAxis::D1)
        .value("D2", RodMaterial::StiffAxis::D2)
        ;

    py::class_<RodMaterial>(m, "RodMaterial")
        .def(py::init<const std::string &, RodMaterial::StiffAxis, bool>(),
                py::arg("cross_section_path.json"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def(py::init<const std::string &, Real, Real, const std::vector<Real> &, RodMaterial::StiffAxis, bool>(),
                py::arg("type"), py::arg("E"), py::arg("nu"),py::arg("params"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def(py::init<const CrossSection &, RodMaterial::StiffAxis, bool>(), py::arg("cs"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def(py::init<>())
        .def("set", py::overload_cast<const std::string &, Real, Real, const std::vector<Real> &, RodMaterial::StiffAxis, bool>(&RodMaterial::set),
                py::arg("type"), py::arg("E"), py::arg("nu"),py::arg("params"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def("setEllipse", &RodMaterial::setEllipse, "Set elliptical cross section")
        .def("setContour", &RodMaterial::setContour, "Set using a custom profile whose boundary is read from a line mesh file",
                py::arg("E"), py::arg("nu"), py::arg("path"), py::arg("scale") = 1.0, py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false, py::arg("debug_psi_path") = std::string(), py::arg("triArea") = 0.001, py::arg("simplifyVisualizationMesh") = 0)
        .def_readwrite("area",                      &RodMaterial::area)
        .def_readwrite("stretchingStiffness",       &RodMaterial::stretchingStiffness)
        .def_readwrite("twistingStiffness",         &RodMaterial::twistingStiffness)
        .def_readwrite("bendingStiffness",          &RodMaterial::bendingStiffness)
        .def_readwrite("momentOfInertia",           &RodMaterial::momentOfInertia)
        .def_readwrite("torsionStressCoefficient",  &RodMaterial::torsionStressCoefficient)
        .def_readwrite("youngModulus",              &RodMaterial::youngModulus)
        .def_readwrite("shearModulus",              &RodMaterial::shearModulus)
        .def_readwrite("crossSectionHeight",        &RodMaterial::crossSectionHeight)
        .def_readwrite("crossSectionBoundaryPts",   &RodMaterial::crossSectionBoundaryPts,   py::return_value_policy::reference)
        .def_readwrite("crossSectionBoundaryEdges", &RodMaterial::crossSectionBoundaryEdges, py::return_value_policy::reference)
        .def("crossSection",                        &RodMaterial::crossSection,              py::return_value_policy::reference)
        .def("releaseCrossSectionMesh",             &RodMaterial::releaseCrossSectionMesh)
        .def_property_readonly("crossSectionMesh",  [](const RodMaterial &rmat) { return std::shared_ptr<CrossSectionMesh::Base>(rmat.crossSectionMeshPtr()); })
        .def("bendingStresses", &RodMaterial::bendingStresses, py::arg("curvatureNormal"))
        .def("copy", [](const RodMaterial &mat) { return std::make_unique<RodMaterial>(mat); })
        .def("stressAnalysis", [](const RodMaterial &mat) { mat.stressAnalysis(); return mat.stressAnalysisPtr(); })
        // Convenience accessors for individual bending stiffness/moment of inertia components
        .def_property("B11", [](const RodMaterial &m          ) { return m.bendingStiffness.lambda_1;       },
                             [](      RodMaterial &m, Real val) {        m.bendingStiffness.lambda_1 = val; })
        .def_property("B22", [](const RodMaterial &m          ) { return m.bendingStiffness.lambda_2;       },
                             [](      RodMaterial &m, Real val) {        m.bendingStiffness.lambda_2 = val; })
        .def_property("I11", [](const RodMaterial &m          ) { return m.momentOfInertia. lambda_1;       },
                             [](      RodMaterial &m, Real val) {        m.momentOfInertia. lambda_1 = val; })
        .def_property("I22", [](const RodMaterial &m          ) { return m.momentOfInertia. lambda_2;       },
                             [](      RodMaterial &m, Real val) {        m.momentOfInertia. lambda_2 = val; })
        .def(py::pickle([](const RodMaterial &mat) {
                    return py::make_tuple(mat.area, mat.stretchingStiffness, mat.twistingStiffness,
                                          mat.bendingStiffness, mat.momentOfInertia,
                                          mat.torsionStressCoefficient, mat.youngModulus, mat.shearModulus,
                                          mat.crossSectionHeight,
                                          mat.crossSectionBoundaryPts,
                                          mat.crossSectionBoundaryEdges,
                                          mat.stressAnalysisPtr());
                },
                [](const py::tuple &t) {
                    if (t.size() < 11 || t.size() > 12) throw std::runtime_error("Invalid state!");
                    RodMaterial mat;
                    mat.area                      = t[0 ].cast<Real>();
                    mat.stretchingStiffness       = t[1 ].cast<Real>();
                    mat.twistingStiffness         = t[2 ].cast<Real>();
                    mat.bendingStiffness          = t[3 ].cast<RodMaterial::DiagonalizedTensor>();
                    mat.momentOfInertia           = t[4 ].cast<RodMaterial::DiagonalizedTensor>();
                    mat.torsionStressCoefficient  = t[5 ].cast<Real>();
                    mat.youngModulus              = t[6 ].cast<Real>();
                    mat.shearModulus              = t[7 ].cast<Real>();
                    mat.crossSectionHeight        = t[8 ].cast<Real>();
                    mat.crossSectionBoundaryPts   = t[9 ].cast<CrossSection::AlignedPointCollection>();
                    mat.crossSectionBoundaryEdges = t[10].cast<std::vector<std::pair<size_t, size_t>>>();

                    if (t.size() < 12) return mat;

                    mat.setStressAnalysisPtr(t[11].cast<std::shared_ptr<CSA>>());

                    return mat;
                }))
        ;
    py::class_<RodMaterial::DiagonalizedTensor>(m, "DiagonalizedTensor")
        .def_readwrite("lambda_1", &RodMaterial::DiagonalizedTensor::lambda_1)
        .def_readwrite("lambda_2", &RodMaterial::DiagonalizedTensor::lambda_2)
        .def("trace", &RodMaterial::DiagonalizedTensor::trace)
        .def(py::pickle([](const RodMaterial::DiagonalizedTensor &d) { return py::make_tuple(d.lambda_1, d.lambda_2); },
                        [](const py::tuple &t) {
                            if (t.size() != 2) throw std::runtime_error("Invalid state!");
                            RodMaterial::DiagonalizedTensor result;
                            result.lambda_1 = t[0].cast<Real>();
                            result.lambda_2 = t[1].cast<Real>();
                            return result;
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // RectangularBoxCollection
    ////////////////////////////////////////////////////////////////////////////////
    auto rectangular_box_collection = py::class_<RectangularBoxCollection>(m, "RectangularBoxCollection")
        .def(py::init<std::vector<RectangularBoxCollection::Corners>>(), py::arg("box_corners"))
        .def(py::init<const std::string>(), py::arg("path"))
        .def("contains", &RectangularBoxCollection::contains, py::arg("p"))
        .def("visualizationGeometry", &getVisualizationGeometry<RectangularBoxCollection>)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Equilibrium solver
    ////////////////////////////////////////////////////////////////////////////////
    m.attr("TARGET_ANGLE_NONE") = py::float_(TARGET_ANGLE_NONE);

    m.def("compute_equilibrium",
          [](ElasticRod &rod, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              return compute_equilibrium(rod, options, fixedVars, cb);
          },
          py::arg("rod"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr
    );

    m.def("get_equilibrium_optimizer",
          [](ElasticRod &rod, const std::vector<size_t> &fixedVars) { return get_equilibrium_optimizer(rod, fixedVars); },
          py::arg("rod"),
          py::arg("fixedVars") = std::vector<size_t>()
    );


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

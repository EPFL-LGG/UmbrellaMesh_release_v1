#include <MeshFEM/MSHFieldWriter.hh>
#include "UmbrellaMesh.hh"
#include <memory>

////////////////////////////////////////////////////////////////////////////////
// I/O for Visualization/Debugging
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
void UmbrellaMesh_T<Real_>::coloredVisualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                                                       std::vector<MeshIO::IOElement> &quads,
                                                       const bool averagedMaterialFrames,
                                                       const bool averagedCrossSections,
                                                       Eigen::VectorXd *height) const {
    for (const auto &s : m_segments)
        s.rod.coloredVisualizationGeometry(vertices, quads, averagedMaterialFrames, averagedCrossSections, height);
}
template<typename Real_>
void UmbrellaMesh_T<Real_>::saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames) const {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> quads;
    visualizationGeometry(vertices, quads, averagedMaterialFrames);
    MeshIO::save(path, vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);
}


////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation for ordinary double type and autodiff type.
////////////////////////////////////////////////////////////////////////////////
template struct UmbrellaMesh_T<double>;
template struct UmbrellaMesh_T<ADReal>;

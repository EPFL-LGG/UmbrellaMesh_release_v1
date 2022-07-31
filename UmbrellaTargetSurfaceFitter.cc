#include "UmbrellaTargetSurfaceFitter.hh"
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Parallelism.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_simplex_squared_distance.h>
#include <igl/AABB.h>
#include <fstream>


struct TargetSurfaceAABB : public igl::AABB<Eigen::MatrixXd, 3> {
    using Base = igl::AABB<Eigen::MatrixXd, 3>;
    using Base::Base;
};

#include <MeshFEM/TriMesh.hh>
struct TargetSurfaceMesh : public TriMesh<> {
    using TriMesh<>::TriMesh;
};


BoundaryEdgeFitter::BoundaryEdgeFitter(const FEMMesh<2, 1, Vec3_T<Real> > &mesh) {
    boundaryEdges.reserve(mesh.numBoundaryEdges());
    for (const auto &be : mesh.boundaryElements()) {
        boundaryEdges.emplace_back(be.node(0).volumeNode()->p,
                                    be.node(1).volumeNode()->p);
    }
}

TargetSurfaceFitter::SurfaceData::SurfaceData(const Eigen::MatrixXd &V_in, const Eigen::MatrixXi &F_in)
    : V(V_in), F(F_in) {

    igl::per_face_normals  (V, F, N);
    igl::per_vertex_normals(V, F, VN);

    std::vector<MeshIO::IOVertex > vertices = getMeshIOVertices(V);
    std::vector<MeshIO::IOElement> elements = getMeshIOElements(F);
    mesh = std::make_unique<TargetSurfaceMesh>(elements, vertices.size());

    aabb_tree = std::make_unique<TargetSurfaceAABB>();
    aabb_tree->init(V, F);
}

template<typename Real_>
void TargetSurfaceFitter::setTargetSurface(const UmbrellaMesh_T<Real_> &umbrella, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    m_tgt_surf = std::make_shared<SurfaceData>(V, F);
    
    std::vector<MeshIO::IOVertex > vertices = getMeshIOVertices(m_tgt_surf->V);
    std::vector<MeshIO::IOElement> elements = getMeshIOElements(m_tgt_surf->F);
    FEMMesh<2, 1, Vec3_T<Real>> tgt_mesh(elements, vertices);
    m_bdryEdgeFitter = BoundaryEdgeFitter(tgt_mesh);

    forceUpdateClosestPoints(umbrella);
}

template<typename Real_>
void TargetSurfaceFitter::loadTargetSurface(const UmbrellaMesh_T<Real_> &umbrella, const std::string &path) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(path, vertices, elements);
    std::cout << "Loaded " << vertices.size() << " vertices and " << elements.size() << " triangles" << std::endl;
    setTargetSurface(umbrella, ::getV(vertices), ::getF(elements));
}

void TargetSurfaceFitter::setBdryEdgeFitter() {
    std::vector<MeshIO::IOVertex > vertices = getMeshIOVertices(m_tgt_surf->V);
    std::vector<MeshIO::IOElement> elements = getMeshIOElements(m_tgt_surf->F);
    FEMMesh<2, 1, Vec3_T<Real>> tgt_mesh(elements, vertices);
    m_bdryEdgeFitter = BoundaryEdgeFitter(tgt_mesh);
}


template<typename Real_>
void TargetSurfaceFitter::forceUpdateClosestPoints(const UmbrellaMesh_T<Real_> &umbrella) {
    if (!m_tgt_surf->aabb_tree) { std::cout << " no aabb" << std::endl; return; }
    const size_t numSamplePts = numQueryPt(umbrella);

    // If we have nonzero weights in the surface-fitting term,
    // or if the closest point array is uninitialized,
    // update each joint's closest surface point.
    if ((size_t(umbrella_closest_surf_pts.size()) == 3 * numSamplePts) && (Wsurf_diag_umbrella_sample_pos.norm() == 0.0)) return;

    std::vector<bool> queryPtIsBoundary = umbrella.IsQueryPtBoundary();
    if (queryPtIsBoundary.size() != numSamplePts) throw std::runtime_error("Invalid isBoundary array");
    
    BENCHMARK_SCOPED_TIMER_SECTION timer("Update closest points");
    umbrella_closest_surf_pts.resize(3 * numSamplePts);
    umbrella_closest_surf_pt_sensitivities.resize(numSamplePts);
    umbrella_closest_surf_tris.resize(numSamplePts);

    int numInterior = 0, numBdryEdge = 0, numBdryVtx = 0;

    parallel_for_range(numSamplePts, [&](size_t pt_i) {



        if (queryPtIsBoundary[pt_i]) {
            Real lambda = 0.0;
            size_t closestEdge = 0;
            Vec3 p, query;
            query = getQueryPtPos(umbrella, pt_i).transpose();
            m_bdryEdgeFitter.closestBarycoordsAndPt(query, lambda, p, closestEdge);
            umbrella_closest_surf_pts.segment<3>(3 * pt_i) = p.transpose();

            if ((lambda == 0.0) || (lambda == 1.0))
                umbrella_closest_surf_pt_sensitivities[pt_i].setZero();
            else {
                const auto &e = m_bdryEdgeFitter.edge(closestEdge).e;
                umbrella_closest_surf_pt_sensitivities[pt_i] = e * e.transpose();
            }
            umbrella_closest_surf_tris[pt_i] = closestEdge;

            return;
        }


        int closest_idx;
        Eigen::RowVector3d p, query;
        query = getQueryPtPos(umbrella, pt_i).transpose();

        Real sqdist = m_tgt_surf->aabb_tree->squared_distance(m_tgt_surf->V, m_tgt_surf->F, query, closest_idx, p);
        umbrella_closest_surf_pts.segment<3>(3 * pt_i) = p.transpose();
        umbrella_closest_surf_tris[pt_i] = closest_idx;

        // Compute the sensitivity of the closest point projection with respect to the query point (dp_dx).
        // There are three cases depending on whether the closest point lies in the target surface's
        // interior, on one of its boundary edges, or on a boundary vertex.
        Eigen::RowVector3d barycoords;
        igl::point_simplex_squared_distance<3>(query,
                                               m_tgt_surf->V, m_tgt_surf->F, closest_idx,
                                               sqdist, p, barycoords);

        std::array<int, 3> boundaryNonzeroLoc;
        int numNonzero = 0, numBoundaryNonzero = 0;
        for (int i = 0; i < 3; ++i) {
            if (barycoords[i] == 0.0) continue;
            ++numNonzero;
            // It is extremely unlikely a vertex will be closest to a point/edge if this is not a stable association.
            // Therefore we assume even for smoothish surfaces that points are constrained to lie on their closest
            // simplex.
            // Hack away the old boundry-snapping-only behavior: treat all non-boundary edges/vertices as active too...
            // TODO: decide on this!
            // if (m_tgt_surf->mesh->vertex(m_tgt_surf->F(closest_idx, i)).isBoundary())
                boundaryNonzeroLoc[numBoundaryNonzero++] = i;
        }
        assert(numNonzero >= 1);

        if ((numNonzero == 3) || (numNonzero != numBoundaryNonzero)) {
            // If the closest point lies in the interior, the sensitivity is (I - n n^T) (the query point perturbation is projected onto the tangent plane).
            umbrella_closest_surf_pt_sensitivities[pt_i] = Eigen::Matrix3d::Identity() - m_tgt_surf->N.row(closest_idx).transpose() * m_tgt_surf->N.row(closest_idx);
            ++numInterior;
        }
        else if ((numNonzero == 2) && (numBoundaryNonzero == 2)) {
            // If the closest point lies on a boundary edge, we assume it can only slide along this edge (i.e., the constraint is active)
            // (The edge orientation doesn't matter.)
            Eigen::RowVector3d e = m_tgt_surf->V.row(m_tgt_surf->F(closest_idx, boundaryNonzeroLoc[0])) -
                                   m_tgt_surf->V.row(m_tgt_surf->F(closest_idx, boundaryNonzeroLoc[1]));
            e.normalize();
            umbrella_closest_surf_pt_sensitivities[pt_i] = e.transpose() * e;
            ++numBdryEdge;
        }
        else if ((numNonzero == 1) && (numBoundaryNonzero == 1)) {
            // If the closest point coincides with a boundary vertex, we assume it is "stuck" there (i.e., the constraint is active)
            umbrella_closest_surf_pt_sensitivities[pt_i].setZero();
            ++numBdryVtx;
        }
        else {
            assert(false);
        }
    });
}


// Visualization functions
std::vector<Real> TargetSurfaceFitter::get_squared_distance_to_target_surface(Eigen::VectorXd query_point_list) const {
    std::vector<Real> output(query_point_list.size() / 3);
    for (size_t pt_i = 0; pt_i < size_t(query_point_list.size()/3); ++pt_i) {
        int closest_idx;
        // Could be parallelized (libigl does this internally for multi-point queries)
        Eigen::RowVector3d p, query;
        query = query_point_list.segment<3>(pt_i * 3).transpose();

        Real sqdist = m_tgt_surf->aabb_tree->squared_distance(m_tgt_surf->V, m_tgt_surf->F, query, closest_idx, p);
        output[pt_i] = sqdist;
    }
    return output;
}

Eigen::VectorXd TargetSurfaceFitter::get_closest_point_for_visualization(Eigen::VectorXd query_point_list) const {
    Eigen::VectorXd output(query_point_list.size());
    for (size_t pt_i = 0; pt_i < size_t(query_point_list.size()/3); ++pt_i) {
        int closest_idx;
        // Could be parallelized (libigl does this internally for multi-point queries)
        Eigen::RowVector3d p, query;
        query = query_point_list.segment<3>(pt_i * 3).transpose();
        m_tgt_surf->aabb_tree->squared_distance(m_tgt_surf->V, m_tgt_surf->F, query, closest_idx, p);
        output.segment<3>(3 * pt_i) = p.transpose();
    }
    return output;
}

Eigen::VectorXd TargetSurfaceFitter::get_closest_point_normal(Eigen::VectorXd query_point_list) {
    Eigen::VectorXd output(query_point_list.size());

    for (size_t pt_i = 0; pt_i < size_t(query_point_list.size()/3); ++pt_i) {
        int closest_idx;
        // Could be parallelized (libigl does this internally for multi-point queries)
        Eigen::RowVector3d p, query, barycoords;

        query = query_point_list.segment<3>(pt_i * 3).transpose();
        Real sqdist = m_tgt_surf->aabb_tree->squared_distance(m_tgt_surf->V, m_tgt_surf->F, query, closest_idx, p);
        igl::point_simplex_squared_distance<3>(query,
                                               m_tgt_surf->V, m_tgt_surf->F, closest_idx,
                                               sqdist, p, barycoords);
        Eigen::Vector3d interpolated_normal(0, 0, 0);
        for (int i = 0; i < 3; ++i) interpolated_normal += barycoords[i] * m_tgt_surf->VN.row(m_tgt_surf->F(closest_idx, i));
        interpolated_normal.normalized();
        output.segment<3>(3 * pt_i) = interpolated_normal;
    }
    return output;
}

TargetSurfaceFitter:: TargetSurfaceFitter() = default;
TargetSurfaceFitter::~TargetSurfaceFitter() = default;

template void TargetSurfaceFitter::forceUpdateClosestPoints<  Real>(const UmbrellaMesh_T<  Real> &umbrella); // explicit instantiation.
template void TargetSurfaceFitter::forceUpdateClosestPoints<ADReal>(const UmbrellaMesh_T<ADReal> &umbrella); // explicit instantiation.
template void TargetSurfaceFitter:: setTargetSurface<ADReal>(const UmbrellaMesh_T<ADReal> &umbrella, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
template void TargetSurfaceFitter:: setTargetSurface<  Real>(const UmbrellaMesh_T<  Real> &umbrella, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
template void TargetSurfaceFitter::loadTargetSurface<ADReal>(const UmbrellaMesh_T<ADReal> &umbrella, const std::string &path);
template void TargetSurfaceFitter::loadTargetSurface<  Real>(const UmbrellaMesh_T<  Real> &umbrella, const std::string &path);

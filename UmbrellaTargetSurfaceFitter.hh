////////////////////////////////////////////////////////////////////////////////
// TargetSurfaceFitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implementation of a target surface to which points are fit using the
//  distance to their closest point projections.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  01/02/2019 11:51:11
////////////////////////////////////////////////////////////////////////////////
#ifndef UMBRELLATARGETSURFACEFITTER_HH
#define UMBRELLATARGETSURFACEFITTER_HH

#include "UmbrellaMesh.hh"
#include "umbrella_compute_equilibrium.hh"
#include <MeshFEM/Utilities/MeshConversion.hh>

#include <MeshFEM/FEMMesh.hh>

// Forward declare mesh data structure for holding the target surface (to avoid bringing in MeshFEM::TriMesh when unnecessary)
struct TargetSurfaceMesh;
// Forward declare AABB data structure
struct TargetSurfaceAABB;


struct BoundaryEdgeFitter {
    using Vec3   = Vec3_T<Real>;
    struct BoundaryEdge {
        BoundaryEdge(Eigen::Ref<const Vec3> p0, Eigen::Ref<const Vec3> p1) {
            e_div_len = p1 - p0;
            Real sqLen = e_div_len.squaredNorm();
            e = e_div_len / std::sqrt(sqLen);

            e_div_len /= sqLen;
            p0_dot_e_div_len = p0.dot(e_div_len);
            endpoints[0] = p0;
            endpoints[1] = p1;
        }

        Real            barycoords(Eigen::Ref<const Vec3> q) const { return q.dot(e_div_len) - p0_dot_e_div_len; }
        Real closestEdgeBarycoords(Eigen::Ref<const Vec3> q) const { return std::max<Real>(std::min<Real>(barycoords(q), 1.0), 0.0); }

        void closestBarycoordsAndPt(Eigen::Ref<const Vec3> q, Real &lambda, Eigen::Ref<Vec3> p) {
            lambda = closestEdgeBarycoords(q);
            p = (1.0 - lambda) * endpoints[0] + lambda * endpoints[1];
        }

        Vec3 e_div_len, e;
        Real p0_dot_e_div_len;
        std::array<Vec3, 2> endpoints;
    };

    BoundaryEdgeFitter() { }

    BoundaryEdgeFitter(const FEMMesh<2, 1, Vec3_T<Real> > &mesh);

    

    void closestBarycoordsAndPt(Eigen::Ref<const Vec3> q, Real &lambda, Eigen::Ref<Vec3> p, size_t &closestEdge) {
        Real sqDist = std::numeric_limits<double>::max();
        const size_t nbe = boundaryEdges.size();
        
        for (size_t i = 0; i < nbe; ++i) {
            Vec3 pp;
            Real l;
            boundaryEdges[i].closestBarycoordsAndPt(q, l, pp);
            Real candidate_sqDist = (q - pp).squaredNorm();
            if (candidate_sqDist < sqDist) {
                sqDist      = candidate_sqDist;
                closestEdge = i;
                p           = pp;
                lambda      = l;
            }
        }
        if (sqDist == std::numeric_limits<double>::max()) throw std::runtime_error("No closest edge found");
    }

    const BoundaryEdge &edge(size_t i) const { return boundaryEdges.at(i); }

    std::vector<BoundaryEdge> boundaryEdges;
};


struct TargetSurfaceFitter {
    using VecX   = VecX_T<Real>;
    using Vec3   = Vec3_T<Real>;
    using Mat3   = Mat3_T<Real>;

    struct SurfaceData {
        SurfaceData(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
        Eigen::MatrixXd V,    // Vertex positions
                        N,    // Per-tri normals
                        VN;   // Per-vertex normals
        Eigen::MatrixXi F;    // Triangle corner vertex indices
        std::shared_ptr<TargetSurfaceAABB> aabb_tree; // Acceleration data structure for closest point queries
        std::shared_ptr<TargetSurfaceMesh> mesh;      // half-edge data structure

        using State = std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>;
        static State serialize(const SurfaceData &data) {
            return std::make_tuple(data.V, data.F);
        }

        static std::shared_ptr<SurfaceData> deserialize(const State &state) {
            return std::make_shared<SurfaceData>(std::get<0>(state), std::get<1>(state));
        }
    };

    TargetSurfaceFitter();
    TargetSurfaceFitter(const TargetSurfaceFitter &tsf) { *this = tsf; }

    // Update the closest points regardless of `holdClosestPointsFixed`
    template<typename Real_>
    void forceUpdateClosestPoints(const UmbrellaMesh_T<Real_> &umbrella);

    template<typename Real_>
    void loadTargetSurface(const UmbrellaMesh_T<Real_> &umbrella, const std::string &path);

    void setBdryEdgeFitter();

    TargetSurfaceFitter &operator=(const TargetSurfaceFitter &tsf) {
        // Copy weights and settings
        W_diag_joint_pos               = tsf.W_diag_joint_pos;
        Wsurf_diag_umbrella_sample_pos = tsf.Wsurf_diag_umbrella_sample_pos;
        holdClosestPointsFixed         = tsf.holdClosestPointsFixed;

        // Copy target/closest point info
        query_pt_pos_tgt                       = tsf.query_pt_pos_tgt;
        umbrella_closest_surf_pts              = tsf.umbrella_closest_surf_pts;
        umbrella_closest_surf_pt_sensitivities = tsf.umbrella_closest_surf_pt_sensitivities;
        umbrella_closest_surf_tris             = tsf.umbrella_closest_surf_tris;

        // Share the target surface instance
        m_tgt_surf = tsf.m_tgt_surf;

        return *this;
    }

    const Eigen::MatrixXd &getV()  const { return m_tgt_surf->V;  }
    const Eigen::MatrixXi &getF()  const { return m_tgt_surf->F;  }
    const Eigen::MatrixXd &getN()  const { return m_tgt_surf->N;  }
    const Eigen::MatrixXd &getVN() const { return m_tgt_surf->VN; }

    template<class Real_>
    size_t numQueryPt(const UmbrellaMesh_T<Real_> &umbrella) const {
        return umbrella.numXJoints() + umbrella.numUmbrellas();
    }
    template<class Real_>
    Vec3 getUmbrellaMidpoint(const UmbrellaMesh_T<Real_> &umbrella, size_t ui) const {
        return 0.5 * (stripAutoDiff(umbrella.joint(umbrella.getUmbrellaCenterJi(ui, 0)).pos()) + stripAutoDiff(umbrella.joint(umbrella.getUmbrellaCenterJi(ui, 1)).pos()));
    }

    template<class Real_>
    Vec3 getQueryPtPos(const UmbrellaMesh_T<Real_> &umbrella, size_t qi) const {
        if (qi < umbrella.numXJoints()) return stripAutoDiff(umbrella.joint(umbrella.get_X_joint_indice_at(qi)).pos());
        if (qi < numQueryPt(umbrella)) return getUmbrellaMidpoint(umbrella, qi - umbrella.numXJoints());
        throw std::runtime_error("Query index out of range!");
    }

    template<class Real_>
    VecX getQueryPtPos(const UmbrellaMesh_T<Real_> &umbrella) const {
        VecX result(numQueryPt(umbrella) * 3);
        for (size_t qi = 0; qi < numQueryPt(umbrella); ++qi) {
            result.template segment<3>(3 * qi) = getQueryPtPos(umbrella, qi);
        }
        return result;
    }

    template<class Real_>
    bool isQueryIndexFeature(const UmbrellaMesh_T<Real_> &umbrella, size_t qi, const std::vector<size_t> &feature_pts) const {
        if (feature_pts.size() == 0) return false;
        if (qi < umbrella.numXJoints()) return (std::find(feature_pts.begin(), feature_pts.end(), umbrella.get_X_joint_indice_at(qi)) != feature_pts.end());
        if (qi < numQueryPt(umbrella)) {
            size_t top_ji = umbrella.getUmbrellaCenterJi(qi - umbrella.numXJoints(), 0);
            size_t bot_ji = umbrella.getUmbrellaCenterJi(qi - umbrella.numXJoints(), 1);
            if ((std::find(feature_pts.begin(), feature_pts.end(), top_ji) != feature_pts.end())) return true;
            return (std::find(feature_pts.begin(), feature_pts.end(), bot_ji) != feature_pts.end());
        }
        throw std::runtime_error("Query index out of range!");
    }

    template<class Real_>
    std::tuple<std::vector<size_t>, Real> getUmbrellaDoFOffsetAndWeightForQueryPt(const UmbrellaMesh_T<Real_> &umbrella, size_t qi) const {
        std::vector<size_t> dof_offset;
        if (qi < umbrella.numXJoints()) {
            dof_offset.push_back(umbrella.dofOffsetForJoint(umbrella.get_X_joint_indice_at(qi)));
            return std::make_tuple(dof_offset, 1.0);
        }
        if (qi < numQueryPt(umbrella)) {
            dof_offset.push_back(umbrella.dofOffsetForJoint(umbrella.getUmbrellaCenterJi(qi - umbrella.numXJoints(), 0)));
            dof_offset.push_back(umbrella.dofOffsetForJoint(umbrella.getUmbrellaCenterJi(qi - umbrella.numXJoints(), 1)));
            return std::make_tuple(dof_offset, 0.5);
        }
        throw std::runtime_error("Query index out of range!");
    }

    template<class Real_>
    Real objective(const UmbrellaMesh_T<Real_> &umbrella) const {
        VecX projectionQueries = getQueryPtPos(umbrella);

        Eigen::VectorXd jointPosDiff = projectionQueries - query_pt_pos_tgt;
        Eigen::VectorXd surfumbrellaSamplePosDiff = projectionQueries - umbrella_closest_surf_pts;
        return 0.5 * (jointPosDiff.dot(W_diag_joint_pos.cwiseProduct(jointPosDiff)) +
                      surfumbrellaSamplePosDiff.dot(Wsurf_diag_umbrella_sample_pos.cwiseProduct(surfumbrellaSamplePosDiff)));
    }

    // Gradient with respect to the query points
    // Note: for the closest point projection term, this gradient expression assumes all components of a query point's
    // weight vector are equal; otherwise the dP/dx expression does not vanish
    // and we need more derivatives of the closest point projection (envelope theorem no longer applies)!
    template<class Real_>
    VecX gradient(const UmbrellaMesh_T<Real_> &umbrella) const {
        VecX projectionQueries = getQueryPtPos(umbrella);
        return m_apply_W(umbrella, projectionQueries - query_pt_pos_tgt) + m_apply_Wsurf(umbrella, projectionQueries - umbrella_closest_surf_pts);
    }

    // More efficient implementation of `gradient` that avoids multiple memory allocations.
    template<class Real_>
    void accumulateGradient(const UmbrellaMesh_T<Real_> &umbrella, typename UmbrellaMesh_T<Real_>::VecX &result, Real weight) const {
        for (size_t qi = 0; qi < numQueryPt(umbrella); ++qi) {
            std::tuple<std::vector<size_t>, Real> offset_and_weight = getUmbrellaDoFOffsetAndWeightForQueryPt(umbrella, qi);
            std::vector<size_t> offsets = std::get<0>(offset_and_weight);
            Real query_weight = std::get<1>(offset_and_weight);
            const auto &pos = getQueryPtPos(umbrella, qi);
            for (size_t ji = 0; ji < offsets.size(); ++ji) {
                result.template segment<3>(offsets[ji]) += weight * query_weight *
                    (             W_diag_joint_pos.segment<3>(3 * qi).cwiseProduct(pos -             query_pt_pos_tgt.segment<3>(3 * qi))
                + Wsurf_diag_umbrella_sample_pos.segment<3>(3 * qi).cwiseProduct(pos - umbrella_closest_surf_pts.segment<3>(3 * qi)));
            }
        }
    }

    // Hessian with respect to query point "vi"
    Mat3 pt_project_hess(size_t vi) const {
        if (holdClosestPointsFixed) return Wsurf_diag_umbrella_sample_pos.segment<3>(vi * 3).asDiagonal();
        // Note: we must assume Wsurf_diag_umbrella_sample_pos.segment<3>(vi * 3) is a multiple of [1, 1, 1], otherwise
        // the true Hessian will require second derivatives of the closest point projection (envelope theorem no longer applies).
        return Wsurf_diag_umbrella_sample_pos[vi * 3] * (Mat3::Identity() - umbrella_closest_surf_pt_sensitivities[vi]);
    }

    Mat3 pt_tgt_hess(size_t vi) const { return W_diag_joint_pos.segment<3>(vi * 3).asDiagonal(); }

    template<class RL_>
    typename RL_::VecX applyHessian(const RL_ &umbrella, Eigen::Ref<const typename RL_::VecX> delta_xp) const {
        using VXd = typename RL_::VecX;
        using V3d = typename RL_::Vec3;

        VXd result = VXd::Zero(delta_xp.size());

        size_t nqp = numQueryPt(umbrella);
        for (size_t qi = 0; qi < nqp; ++qi) {
            std::tuple<std::vector<size_t>, Real> offset_and_weight = getUmbrellaDoFOffsetAndWeightForQueryPt(umbrella, qi);
            std::vector<size_t> offsets = std::get<0>(offset_and_weight);
            Real dof_weight = std::get<1>(offset_and_weight);
            V3d delta_pt = delta_xp.template segment<3>(offsets[0]);
            for (size_t oi = 1; oi < offsets.size(); ++oi) delta_pt += delta_xp.template segment<3>(offsets[oi]);
            delta_pt *= dof_weight;

            // Hessian of the surface projection term
            V3d delta_grad = pt_project_hess(qi) * delta_pt;
            // Hessian of the surface projection term
            delta_grad += W_diag_joint_pos.segment<3>(qi * 3).asDiagonal() * delta_pt;

            for (size_t o : offsets)
                result.template segment<3>(o) += dof_weight * delta_grad;
        };

        return result;
    }

    template<typename Real_> void constructTargetSurface(const UmbrellaMesh_T<Real_> &umbrella, size_t loop_subdivisions = 0);
                             void setTargetSurfaceHelper(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
    template<typename Real_> void setTargetSurface(const UmbrellaMesh_T<Real_> &umbrella, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
    template<typename Real_> void updateClosestPoints(const UmbrellaMesh_T<Real_> &umbrella) {
        if (!holdClosestPointsFixed) {
            forceUpdateClosestPoints(umbrella);
        }
    }

    template<typename Real_>
    void scaleJointWeights(const UmbrellaMesh_T<Real_> &umbrella, Real jointPosWeight, Real bdryMultiplier = 1.0, Real featureMultiplier = 1.0, const std::vector<size_t> &additional_feature_pts = std::vector<size_t>(), bool updateClosestPoints = true) {
        // Given the valence 2 vertices a valence2Multiplier times higher weight for fitting to their target positions.
        // (But leave the target surface fitting weights uniform).
        const size_t nqp = numQueryPt(umbrella);

        std::vector<bool> queryPtIsBoundary = umbrella.IsQueryPtBoundary();
        
        
        size_t nbqp = 0;
        for (size_t qi = 0; qi < nqp; ++qi) if(queryPtIsBoundary[qi]) nbqp++;
        const size_t niqp = nqp - nbqp;
        // For now, features and non-features are only interior
        size_t numFeatures = additional_feature_pts.size();
        size_t numNonFeatures = niqp - numFeatures;
        Real nonFeatureWeight = 1.0 / (3.0 * (numFeatures * featureMultiplier + numNonFeatures));
        
        size_t numJointPosComponents = 3 * nqp;
        W_diag_joint_pos.resize(numJointPosComponents);
        
        for (size_t qi = 0; qi < nqp; ++qi) {
            if (queryPtIsBoundary[qi]) continue; // for bdry Points we do not care about the given correspondences (since currently they don't necessarily lie on the boundary)
            
            W_diag_joint_pos.segment<3>(3 * qi).setConstant(jointPosWeight * nonFeatureWeight * (isQueryIndexFeature(umbrella, qi, additional_feature_pts) ? featureMultiplier : 1.0) );
        }
            
            
        size_t nsc = 3 * nqp;
        Wsurf_diag_umbrella_sample_pos.resize(nsc);
        for (size_t qi = 0; qi < nqp; ++qi) {
            Real interiorWeight = (1.0 - jointPosWeight) / (3.0 * (niqp + bdryMultiplier*nbqp));
            if(queryPtIsBoundary[qi]) 
                Wsurf_diag_umbrella_sample_pos.segment<3>(3*qi).setConstant(interiorWeight * bdryMultiplier);
            else
                Wsurf_diag_umbrella_sample_pos.segment<3>(3*qi).setConstant(interiorWeight);
        }
        if (updateClosestPoints) forceUpdateClosestPoints(umbrella);
    }

    std::vector<Real> get_squared_distance_to_target_surface(Eigen::VectorXd query_point_list) const;
    Eigen::VectorXd get_closest_point_for_visualization(Eigen::VectorXd query_point_list) const;
    Eigen::VectorXd get_closest_point_normal(Eigen::VectorXd query_point_list);

    template<typename Real_>
    void reset_joint_target_with_closest_points(const UmbrellaMesh_T<Real_> &umbrella) {
        forceUpdateClosestPoints(umbrella);
        if (query_pt_pos_tgt.size() != umbrella_closest_surf_pts.size()) throw std::runtime_error("The number of target joint positions doesn't match the number of closest point query!");
        query_pt_pos_tgt = umbrella_closest_surf_pts;
    }

    ~TargetSurfaceFitter();

private:
    // Apply the joint position weight matrix W to a compressed state vector that
    // contains only variables corresponding to joint positions.
    // Returns an uncompressed vector with an entry for each state variable.
    template<typename Real_> Eigen::VectorXd m_apply_W    (const UmbrellaMesh_T<Real_> &umbrella, const Eigen::Ref<const Eigen::VectorXd> &queryPtPos) const { return m_unpackQueryPtGradient(umbrella, W_diag_joint_pos.cwiseProduct(queryPtPos)); }
    template<typename Real_> Eigen::VectorXd m_apply_Wsurf(const UmbrellaMesh_T<Real_> &umbrella, const Eigen::Ref<const Eigen::VectorXd> &x_sample_point_pos) const {
        auto weighted_sample_point_pos = Wsurf_diag_umbrella_sample_pos.cwiseProduct(x_sample_point_pos);
        return m_unpackQueryPtGradient(umbrella, weighted_sample_point_pos);
    }
    // Extract a full state vector from a compressed version that only holds
    // variables corresponding to query point positions.
    template<typename Real_>
    Eigen::VectorXd m_unpackQueryPtGradient(const UmbrellaMesh_T<Real_> &umbrella, const Eigen::Ref<const Eigen::VectorXd> &queryPtGradient) const {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(umbrella.numDoF());
        for (size_t qi = 0; qi < numQueryPt(umbrella); ++qi) {
            std::tuple<std::vector<size_t>, Real> offset_and_weight = getUmbrellaDoFOffsetAndWeightForQueryPt(umbrella, qi);
            std::vector<size_t> offsets = std::get<0>(offset_and_weight);
            Real query_weight = std::get<1>(offset_and_weight);
            for (size_t ji = 0; ji < offsets.size(); ++ji) {
                result.template segment<3>(offsets[ji]) = query_weight * queryPtGradient.segment<3>(3 * qi);
            }
        }
        return result;
    }
    ////////////////////////////////////////////////////////////////////////////
    // Private member variables
    ////////////////////////////////////////////////////////////////////////////
    // Target surface to which the deployed joints are fit.
    // This data is meant to be shared across several distinct instances of the
    // target surface fitter (e.g., for an UmbrellaOptimization's target fitting
    // term, in its committed umbrella mesh's attraction term, and in its
    // linesearch umbrella mesh's attraction term.)
    std::shared_ptr<SurfaceData> m_tgt_surf;

public:
    ////////////////////////////////////////////////////////////////////////////
    // Serialization + cloning support (for pickling)
    ////////////////////////////////////////////////////////////////////////////
    using State = std::tuple<std::shared_ptr<SurfaceData>,
                             Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, std::vector<Eigen::Matrix3d>, std::vector<int>, bool>;
    static State serialize(const TargetSurfaceFitter &tsf) {
        return std::make_tuple(tsf.m_tgt_surf,
                               tsf.W_diag_joint_pos, tsf.Wsurf_diag_umbrella_sample_pos, tsf.query_pt_pos_tgt, tsf.umbrella_closest_surf_pts, tsf.umbrella_closest_surf_pt_sensitivities, tsf.umbrella_closest_surf_tris, tsf.holdClosestPointsFixed);
    }

    static std::shared_ptr<TargetSurfaceFitter> deserialize(const State &state) {
        auto tsf = std::make_shared<TargetSurfaceFitter>();
        tsf->m_tgt_surf                             = std::get<0>(state);
        tsf->W_diag_joint_pos                       = std::get< 1>(state);
        tsf->Wsurf_diag_umbrella_sample_pos         = std::get< 2>(state);
        tsf->query_pt_pos_tgt                       = std::get< 3>(state);
        tsf->umbrella_closest_surf_pts              = std::get< 4>(state);
        tsf->umbrella_closest_surf_pt_sensitivities = std::get< 5>(state);
        tsf->umbrella_closest_surf_tris             = std::get< 6>(state);
        tsf->holdClosestPointsFixed                 = std::get< 7>(state);

        tsf->setBdryEdgeFitter();

        return tsf;
    }



    std::shared_ptr<TargetSurfaceFitter> clone() { return deserialize(serialize(*this)); }

    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    // Fitting weights
    Eigen::VectorXd W_diag_joint_pos,       // compressed version of W from the writeup: only include weights corresponding to joint position variables.
                    Wsurf_diag_umbrella_sample_pos;   // Similar to above, the weights for fitting each joint to its closest point on the surface.
                                            // WARNING: if this is changed from zero to a nonzero value, the joint_closest_surf_pts will not be updated
                                            // until the next equilibrium solve.
    Eigen::VectorXd query_pt_pos_tgt;       // compressed, flattened version of x_tgt from the writeup: only include the joint position variables

    Eigen::VectorXd umbrella_closest_surf_pts;                           // compressed, flattened version of p(x)  from the writeup: only include the joint position variables
    std::vector<Eigen::Matrix3d> umbrella_closest_surf_pt_sensitivities; // dp_dx(x) from writeup (sensitivity of closest point projection)
    std::vector<int> umbrella_closest_surf_tris;                         // for debugging: index of the closest triangle to each joint.
    bool holdClosestPointsFixed = false;

    BoundaryEdgeFitter m_bdryEdgeFitter;
};

#endif /* end of include guard: UMBRELLATARGETSURFACEFITTER_HH */

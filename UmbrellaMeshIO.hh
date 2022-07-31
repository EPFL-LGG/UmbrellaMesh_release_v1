#ifndef UMBRELLAMESHIO_HH
#define UMBRELLAMESHIO_HH
#include <MeshFEM/Types.hh>
#include <stdexcept>


// Structure collecting all the input data needed to construct an Umbrella mesh
struct UmbrellaMeshIO {
    using V3d = Eigen::Vector3d;

    enum class JointType {Rigid, X, T};
    enum class JointPosType {Top, Bot, Arm};
    enum class SegmentType {Plate, Arm};
    enum class ArmSegmentPosType {Une, Deux, Trois, Null};
    enum class DeploymentForceType { Spring, Constant, LinearActuator };
    enum class AngleBoundEnforcement { Disable, Hard, Penalty };

    struct Joint {
        Joint(JointType t, const V3d &p, const V3d &b, const V3d &n, double a, const std::vector<size_t> &uid, const V3d corr)
            : type(t), position(p), bisector(b), normal(n), tgt_pos(corr), alpha(a), umbrella_ID(uid) { }

        JointType type;
        V3d position, bisector, normal, tgt_pos;
        double alpha; // Initial opening angle

        std::vector<size_t> umbrella_ID; // Umbrella(s) to which this joint is associated
    };

    // A segment endpoint's connection with a joint
    struct JointConnection {
        JointConnection(size_t ji, bool isA, const V3d &offset)
            : joint_index(ji), is_A(isA), midpoint_offset(offset) { }
        size_t joint_index;   // Global index of the joint
        bool is_A;            // Whether the segment is attached to the joint's "A" or "B" rigid bodies ("ghosts")
        V3d midpoint_offset;  // Offset from the joint position to the segment's terminal edge midpoint
    };

    struct Segment {
        Segment(SegmentType t, const std::vector<JointConnection> &endpt, const V3d &n)
            : type(t), endpoint(endpt), normal(n) { }
        SegmentType type;
        std::vector<JointConnection> endpoint; // Avoiding std::array to work around pybind11 incompatibility
        V3d normal;
    };

    struct Umbrella {
        Umbrella(size_t j_top, size_t j_bot, const V3d corr)
            : top_joint(j_top), bottom_joint(j_bot), tgt_pos(corr) { }
        size_t top_joint, bottom_joint;
        V3d tgt_pos;
    };

    UmbrellaMeshIO(const std::vector<Joint>    &j,
                   const std::vector<Segment>  &s,
                   const std::vector<Umbrella> &u,
                   const std::vector<std::vector<size_t>> &connectivity,
                   const std::vector<double> &m,
                   const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
        : joints(j), segments(s), umbrellas(u), umbrella_connectivity(connectivity), material_params(m), target_v(V), target_f(F)
    { }

    std::vector<Joint>    joints;
    std::vector<Segment>  segments;
    std::vector<Umbrella> umbrellas;
    std::vector<std::vector<size_t>> umbrella_connectivity; // Edge list based on how umbrellas are connected - [uid1, uid2]
    std::vector<double> material_params; // [E1, nu1, thickness1, width1, E2, nu2, thickness2, width2]
    Eigen::MatrixXd target_v;
    Eigen::MatrixXi target_f;

    JointType jointType(size_t ji) const { return joints[ji].type; }

    void validate() const {
        const size_t nj = joints.size();
        const size_t ns = segments.size();
        if (nj <= 0) throw std::runtime_error("There must be at least one joint in the mesh!");

        // Validate the segment-joint incidence
        std::vector<size_t> jointValence(nj);
        for (size_t si = 0; si < ns; ++si) {
            const auto &s = segments[si];
            if (s.endpoint.size() != 2)          throw std::runtime_error("Segment " + std::to_string(si) + " does not have exactly two endpoints.");
            if (s.endpoint[0].joint_index >= nj) throw std::runtime_error("Joint 0 of segment " + std::to_string(si) + " out of bounds");
            if (s.endpoint[1].joint_index >= nj) throw std::runtime_error("Joint 1 of segment " + std::to_string(si) + " out of bounds");
            ++jointValence.at(s.endpoint[0].joint_index);
            ++jointValence.at(s.endpoint[1].joint_index);
        }
        for (size_t ji = 0; ji < nj; ++ji)
            if (jointValence[ji] == 0) throw std::runtime_error("Joint " + std::to_string(ji) + " is dangling");

        ////////////////////////////////////////////////////////////////////////////
        // Check Validity of umbrella_to_top_bottom_joint_map
        ////////////////////////////////////////////////////////////////////////////
        for (const auto &u : umbrellas) {
            if (!(jointType(u.   top_joint) == JointType::Rigid &&
                  jointType(u.bottom_joint) == JointType::Rigid)) throw std::runtime_error("Plate top and bot joints must be rigid");
        }
        ////////////////////////////////////////////////////////////////////////////
        // Check Validity of umbrella_connectivity
        ////////////////////////////////////////////////////////////////////////////
        for (const auto &e : umbrella_connectivity) {
            if (e.size() != 2) throw std::runtime_error("Umbrella connectivity must comprise of edges made of uids");
        }

        if (material_params.size() != 8) throw std::runtime_error("Material params must be [E1, nu1, thickness1, width1, E2, nu2, thickness2, width2]");
    }
};

#endif /* end of include guard: UMBRELLAMESHIO_HH */

#include "UmbrellaRestHeightsOptimization.hh"

void UmbrellaRestHeightsOptimization::m_checkArmHeightValidity(const UmbrellaMesh &um) const {
    for (size_t ai = 0; ai < um.numArmSegments(); ++ai) {
        size_t si = um.getSegmentIndexForArmAt(ai);
        size_t sJoint = um.segment(si).startJoint, eJoint = um.segment(si).endJoint;
        if (um.joint(sJoint).jointType() == UmbrellaMesh::JointType::X) {
            if (um.joint(eJoint).jointType() != UmbrellaMesh::JointType::T) throw std::runtime_error("Segment " + std::to_string(si) + "is an arm segment but doesn't contain one X, and one T joints");
        }
        else if (um.joint(eJoint).jointType() == UmbrellaMesh::JointType::X) {
            if (um.joint(sJoint).jointType() != UmbrellaMesh::JointType::T) throw std::runtime_error("Segment " + std::to_string(si) + "is an arm segment but doesn't contain one X, and one T joints");
        }
        else throw std::runtime_error("One of the joints of an arm segment must be of type X");
    }
}

void UmbrellaRestHeightsOptimization::m_constructUmbrellaRestHeightsToArmRestLenMapTranspose() {    
    auto & um = m_um_opt.committedObject();
    m_checkArmHeightValidity(um);
    const SuiteSparse_long m = um.numURH(), n = um.numArmSegments();
    SuiteSparseMatrix result(m, n);
    result.nz = um.numArmSegments();

    // Now we fill out the transpose of the map one column (arm segment) at a time:
    //    #         [               ]
    // umbrella * 2 [               ]
    //                # arm segment

    size_t numUmbrellas = um.numURH() / 2;
    result.Ax.assign(result.nz, 1);
    auto &Ai = result.Ai;
    auto &Ap = result.Ap;

    Ai.reserve(result.nz);
    Ap.reserve(n + 1);

    Ap.push_back(0); // col 0 begin

    for (size_t ai = 0; ai < um.numArmSegments(); ++ai) {
        size_t si = um.getSegmentIndexForArmAt(ai);
        size_t sJoint = um.segment(si).startJoint, eJoint = um.segment(si).endJoint;
        size_t tJoint = (um.joint(sJoint).jointType() == UmbrellaMesh::JointType::X) ? eJoint : sJoint;
        size_t uid = um.joint(tJoint).umbrellaID()[0]; // Fetch the current umbrella id of t-joint. Index 1 would have the neighbor ID.
        if(um.joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Top) Ai.push_back(uid);
        else if(um.joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Bot) Ai.push_back(numUmbrellas + uid);
        else throw std::runtime_error("tJoint can't be of JointPosType Arm");
        Ap.push_back(Ai.size()); // col end
    }

    assert(Ai.size() == size_t(result.nz));
    assert(Ap.size() == size_t(n + 1    ));

    m_umbrellaRestHeightsToArmRestLenMapTranspose = std::move(result);

    // Verify m_designParametersURH, m_umbrellaRestHeightsToArmRestLenMapTranspose are consistent
    if ((um.getPerArmRestLength() - m_umbrellaRestHeightsToArmRestLenMapTranspose.apply(params(), /* transpose */ true)).norm() > 1e-8)
        throw std::runtime_error("Check consistency of m_designParametersURH, m_umbrellaRestHeightsToArmRestLenMapTranspose");
}

const Eigen::VectorXd UmbrellaRestHeightsOptimization::params() {
    auto & um = m_um_opt.committedObject();
    m_checkArmHeightValidity(um);
    Eigen::VectorXd result(um.numURH());
    result.setZero();
    size_t numUmbrellas = um.numURH() / 2;
    auto parl = um.getPerArmRestLength();
    for (size_t ai = 0; ai < um.numArmSegments(); ++ai) {
        size_t si = um.getSegmentIndexForArmAt(ai);
        size_t sJoint = um.segment(si).startJoint, eJoint = um.segment(si).endJoint;
        size_t tJoint = (um.joint(sJoint).jointType() == UmbrellaMesh::JointType::X) ? eJoint : sJoint;
        size_t uid = um.joint(tJoint).umbrellaID()[0]; // Fetch the current umbrella id of t-joint. Index 1 would have the neighbor ID.
        if(um.joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Top) result(uid) = parl[ai];
        else if(um.joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Bot) result(numUmbrellas + uid) = parl[ai];
    }
    return result;
}

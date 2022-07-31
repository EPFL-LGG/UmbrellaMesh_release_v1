#include "UmbrellaFourParametersOptimization.hh"

void UmbrellaFourParametersOptimization::m_checkArmHeightValidity(const UmbrellaMesh &um) const {
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

void UmbrellaFourParametersOptimization::m_constructUmbrellaFourParametersToArmRestLenMapTranspose() {    
    auto & um = m_um_opt.committedObject();
    m_checkArmHeightValidity(um);
    const SuiteSparse_long m = um.numURH() * 2, n = um.numArmSegments();
    SuiteSparseMatrix result(m, n);
    result.nz = um.numArmSegments() + um.numBottomArmSegments();

    // Now we fill out the transpose of the map one column (arm segment) at a time:
    //    #         [               ]
    // umbrella * 4 [               ]
    //                # arm segment

    size_t numUmbrellas = um.numURH() / 2;
    auto &Ax = result.Ax;
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
        
        size_t umbrella_arm_index = 0;
        if (um.segment(si).armSegmentPosType() != UmbrellaMesh::ArmSegmentPosType::Null) {
            umbrella_arm_index = static_cast<size_t>(um.segment(si).armSegmentPosType());
        } else {
            throw std::runtime_error("Arm segment pos type is NULL for an arm segment!");
        }
        if (um.joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Top) {
            // Need to get the umbrella arm index of this segment
            Ax.push_back(1);
            Ai.push_back(numUmbrellas * umbrella_arm_index + uid);
        } else if(um.joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Bot) {
            // Need to get the umbrella arm index of this segment and assign -1
            Ax.push_back(-1);
            Ai.push_back(numUmbrellas * umbrella_arm_index + uid);
            Ax.push_back(1);
            Ai.push_back(numUmbrellas * 3 + uid);
        }
        else throw std::runtime_error("tJoint can't be of JointPosType Arm");
        Ap.push_back(Ai.size()); // col end
    }

    assert(Ai.size() == size_t(result.nz));
    assert(Ap.size() == size_t(n + 1    ));

    m_umbrellaFourParametersToArmRestLenMapTranspose = std::move(result);

    // Verify m_designParametersURFP, m_umbrellaFourParametersToArmRestLenMapTranspose are consistent
    if ((um.getPerArmRestLength() - m_umbrellaFourParametersToArmRestLenMapTranspose.apply(params(), /* transpose */ true)).norm() > 1e-8)
        throw std::runtime_error("Check consistency of m_designParametersURFP, m_umbrellaFourParametersToArmRestLenMapTranspose");
}

std::vector<size_t> UmbrellaFourParametersOptimization::m_setEmptyArmParams() {
    m_emptyArmParams.clear();
    auto & um = m_um_opt.committedObject();
    m_checkArmHeightValidity(um);
    Eigen::VectorXd result(um.numURH() * 2);
    result.setZero();
    size_t numUmbrellas = um.numURH() / 2;
    for (size_t ai = 0; ai < um.numArmSegments(); ++ai) {
        size_t si = um.getSegmentIndexForArmAt(ai);
        size_t sJoint = um.segment(si).startJoint, eJoint = um.segment(si).endJoint;
        size_t tJoint = (um.joint(sJoint).jointType() == UmbrellaMesh::JointType::X) ? eJoint : sJoint;
        size_t uid = um.joint(tJoint).umbrellaID()[0]; // Fetch the current umbrella id of t-joint. Index 1 would have the neighbor ID.
        size_t umbrella_arm_index = 0;
        if (um.segment(si).armSegmentPosType() != UmbrellaMesh::ArmSegmentPosType::Null) {
            umbrella_arm_index = static_cast<size_t>(um.segment(si).armSegmentPosType());
        } else {
            throw std::runtime_error("Arm segment pos type is NULL for an arm segment!");
        }
        if(um.joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Top) {
            result(numUmbrellas * umbrella_arm_index + uid) = 1;
            if (result(numUmbrellas * 3 + uid) == 0) result(numUmbrellas * 3 + uid) = 1;
        }
    }
    for (size_t ri = 0; ri < numUmbrellas * 4; ++ri) {
        if (result[ri] == 0) m_emptyArmParams.push_back(ri);
    }
    return m_emptyArmParams;
}

const Eigen::VectorXd UmbrellaFourParametersOptimization::params() {
    auto & um = m_um_opt.committedObject();
    m_checkArmHeightValidity(um);
    Eigen::VectorXd result(um.numURH() * 2);
    result.setZero();
    size_t numUmbrellas = um.numURH() / 2;
    auto parl = um.getPerArmRestLength();
    for (size_t ai = 0; ai < um.numArmSegments(); ++ai) {
        size_t si = um.getSegmentIndexForArmAt(ai);
        size_t sJoint = um.segment(si).startJoint, eJoint = um.segment(si).endJoint;
        size_t tJoint = (um.joint(sJoint).jointType() == UmbrellaMesh::JointType::X) ? eJoint : sJoint;
        size_t uid = um.joint(tJoint).umbrellaID()[0]; // Fetch the current umbrella id of t-joint. Index 1 would have the neighbor ID.
        size_t umbrella_arm_index = 0;
        if (um.segment(si).armSegmentPosType() != UmbrellaMesh::ArmSegmentPosType::Null) {
            umbrella_arm_index = static_cast<size_t>(um.segment(si).armSegmentPosType());
        } else {
            throw std::runtime_error("Arm segment pos type is NULL for an arm segment!");
        }
        if(um.joint(tJoint).jointPosType() == UmbrellaMesh::JointPosType::Top) {
            result(numUmbrellas * umbrella_arm_index + uid) = parl[ai];
            // Accumulate the height variables.
            if (result(numUmbrellas * 3 + uid) == 0) result(numUmbrellas * 3 + uid) = parl[ai] + parl[um.getMirrorArm(si)];
        }
    }
    return result;
}

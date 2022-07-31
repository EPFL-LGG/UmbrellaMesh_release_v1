#include "UmbrellaMeshTerminalEdgeSensitivity.hh"
#include "UmbrellaTargetSurfaceFitter.hh"
#include <MeshFEM/ParallelAssembly.hh>
template<typename Real_>
template<class F>
void UmbrellaMesh_T<Real_>::m_assembleSegmentGradient(const F &gradientGetter,
        VecX_T<Real_> &g, bool updatedSource, bool variableDesignParameters,
        bool designParameterOnly) const {
    {
        // Generally when evaluating the gradient at a new iterate (when the source frame is updated),
        // the user will also want to compute the Hessian shortly thereafter.
        // Since the joint parametrization Hessian formula can re-use several
        // values computed for the parametrization Jacobian, for now we always
        // pre-compute and cache the parametrization Hessian. If we find a
        // usage pattern where the gradient is evaluated at many iterates where
        // the Hessian is *not* requested, we might change this strategy.
        const bool evalHessian = updatedSource;
        m_sensitivityCache.update(*this, updatedSource, evalHessian);
    }

    // Accumulate contribution of each segment's elastic energy gradient to the full gradient
    auto accumulateSegment = [&](const size_t si, VecX &gout) {
        const auto &s = m_segments[si];
        auto &r = s.rod;
        // Partial derivatives with respect to the segment's unconstrained DoFs
        const typename ElasticRod_T<Real_>::Gradient &sg = gradientGetter(s);
        const size_t nv = r.numVertices(), ne = r.numEdges();
        // Design parameter derivatives
        if (variableDesignParameters) {
            if (m_umbrella_dPC.restLen) {
                // Copy over the gradient components for the degrees of freedom
                // that directly control interior/free-end edge rest length parameters.
                if (s.segmentType() == SegmentType::Arm) {
                    gout.segment(m_restLenDofOffsetForSegment[si], s.numFreeEdges()) =
                        sg.segment(sg.designParameterOffset + s.hasStartJoint(), s.numFreeEdges());
                    // Accumulate contributions to the rest lengths controlled by each joint
                    for (size_t i = 0; i < 2; ++i) {
                        size_t jindex = s.joint(i);
                        if (jindex == NONE) continue;
                        const auto &joint = m_joints.at(jindex);
                        
                        size_t localSegmentIndex;
                        bool is_A, isStart;
                        std::tie(is_A, isStart, localSegmentIndex) = joint.terminalEdgeIdentification(si);
                        size_t jointRestLengthOffset = joint.arm_offset(localSegmentIndex);
                        size_t dofIdx = m_designParameterDoFOffsetForJoint[jindex] + jointRestLengthOffset;
                        size_t edgeIdx = isStart ? 0 : r.numEdges() - 1;
                        gout[dofIdx] += sg.gradDesignParameters(edgeIdx);
                    }
                }
            }
            if (designParameterOnly) return;        
        }

        size_t offset = m_dofOffsetForSegment[si];

        // Copy over the gradient components for the degrees of freedom that
        // directly control the interior/free-end centerline positions and
        // material frame angles.
        for (size_t i = 0; i < nv; ++i) {
            // The first/last edge don't contribute degrees of freedom if they're part of a joint.
            if ((i <       2) && s.hasStartJoint()) continue;
            if ((i >= nv - 2) && s.  hasEndJoint()) continue;
            gout.template segment<3>(offset) = sg.gradPos(i);
            offset += 3;
        }
        for (size_t j = 0; j < ne; ++j) {
            if ((j ==      0) && s.hasStartJoint()) continue;
            if ((j == ne - 1) && s.  hasEndJoint()) continue;
            gout[offset++] = sg.gradTheta(j);
        }

        // Accumulate contributions to the start/end joints (if they exist)
        for (size_t i = 0; i < 2; ++i) {
            size_t jindex = s.joint(i);
            if (jindex == NONE) continue;

            offset = m_dofOffsetForJoint.at(jindex);
            const auto &sensitivity = m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            const size_t j = sensitivity.j;
            //           pos       e_X  theta^j   p_X
            // x_j     [  I    - 0.5 I     0       I] [ I 0 ... 0]
            // x_{j+1} [  I      0.5 I     0       I] [ jacobian ]
            // theta^j [  0          0     I       0]
            gout.template segment<3>(offset + 0) += sg.gradPos(j) + sg.gradPos(j + 1);

            Eigen::Matrix<Real_, 7, 1> dE_djointvar;
            dE_djointvar.template segment<3>(0) = 0.5 * (sg.gradPos(j + 1) - sg.gradPos(j));
            dE_djointvar[3] = sg.gradTheta(j);
            dE_djointvar.template segment<3>(4) = sg.gradPos(j + 1) + sg.gradPos(j);
            VecX grad_result = sensitivity.jacobian.transpose() * dE_djointvar;
            // Assign the computed gradient in the appropriate location in the global vector.
            // -3 to shift out the position variables.
            gout.segment(offset + 3, joint(jindex).numBaseDoF() - 3) += grad_result.segment(0, joint(jindex).numBaseDoF() - 3);
            const size_t lenVarIndex = offset + joint(jindex).numBaseDoF() + sensitivity.localSegmentIndex;
            gout[lenVarIndex] += grad_result[4];

        }
    };

    assemble_parallel(accumulateSegment, g, numSegments());
}

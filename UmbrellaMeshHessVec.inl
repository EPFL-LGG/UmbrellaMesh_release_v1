#ifndef UMBRELLAMESHHESSVEC_INL
#define UMBRELLAMESHHESSVEC_INL

template<typename Real_>
struct RodHessianApplierData {
    VecX_T<Real_> v_local, Hv_local, Hv;
    bool constructed = false;
};

#if MESHFEM_WITH_TBB
template<typename Real_>
using RHALocalData = tbb::enumerable_thread_specific<RodHessianApplierData<Real_>>;

template<typename F, typename Real_>
struct RodHessianApplier {
    RodHessianApplier(F &f, const size_t nvars, RHALocalData<Real_> &locals) : m_f(f), m_nvars(nvars), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        RodHessianApplierData<Real_> &data = m_locals.local();
        if (!data.constructed) { data.Hv.setZero(m_nvars); data.constructed = true; }
        for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, data); }
    }
private:
    F &m_f;
    size_t m_nvars;
    RHALocalData<Real_> &m_locals;
};

template<typename F, typename Real_>
RodHessianApplier<F, Real_> make_rod_hessian_applier(F &f, size_t nvars, RHALocalData<Real_> &locals) {
    return RodHessianApplier<F, Real_>(f, nvars, locals);
}
#endif

template<typename Real_>
auto UmbrellaMesh_T<Real_>::applyHessianElastic(const VecX &v, bool variableDesignParameters, const HessianComputationMask &mask) const -> VecX {
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".applyHessianElastic");
    const size_t ndof = variableDesignParameters ? numExtendedDoF() : numDoF();
    if (size_t(v.size()) != ndof) throw std::runtime_error("Input vector size mismatch");

    // Our Hessian can only be evaluated after the source configuration has
    // been updated; use the more efficient gradient formulas.
    const bool updatedSource = true;
    {
        const bool hessianNeeded = mask.dof_in && mask.dof_out; // joint parametrization Hessian only needed for dof-dof part
        if (hessianNeeded) m_sensitivityCache.update(*this, updatedSource, v); // directional derivative only
        else               m_sensitivityCache.update(*this, updatedSource, false); // In all cases, we need at least the Jacobian
    }

    // Note: `mask.skipBRods` is a hack to compute derivatives of rivet forces;
    // it effectively detaches the joints from their local B segments (but
    // doesn't actually skip the whole "B segment" since a given segment could
    // be labeled "A" at one joint and "B" at the other.
    auto applyPerSegmentHessian = [&](const size_t si, RodHessianApplierData<Real_> &data) {
        VecX & v_local = data.v_local;
        VecX &Hv_local = data.Hv_local;
        VecX &Hv       = data.Hv;

        const auto &s = m_segments[si];
        const auto &r = s.rod;

        std::array<const UmbrellaMeshTerminalEdgeSensitivity<Real_> *, 2> jointSensitivity{{ nullptr, nullptr }};
        std::array<size_t, 2> segmentJointDofOffset;
        for (size_t i = 0; i < 2; ++i) {
            size_t ji = s.joint(i);
            if (ji == NONE) continue;
            jointSensitivity[i] = &m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            segmentJointDofOffset[i] = m_dofOffsetForJoint[ji];
        }

        const size_t ndof_local = variableDesignParameters ? s.rod.numExtendedDoF() : s.rod.numDoF();
        if (s.rod.numDoF() + s.rod.numDesignParameters() != s.rod.numExtendedDoF()) throw std::runtime_error("DoF count mismatch");
        // Apply dv_dr (compute the perturbation of the rod variables).
        v_local.resize(ndof_local);

        if (mask.dof_in) {
            // Copy over the interior/free-end vertex and theta perturbations.
            const size_t free_vtx_components = 3 * s.numFreeVertices(),
                         local_theta_offset = s.rod.thetaOffset();
            v_local.segment((3 * 2) * s.hasStartJoint(),         free_vtx_components) = v.segment(m_dofOffsetForSegment[si], free_vtx_components);
            v_local.segment(local_theta_offset + s.hasStartJoint(), s.numFreeEdges()) = v.segment(m_dofOffsetForSegment[si] + free_vtx_components, s.numFreeEdges());

            // Compute the perturbations of the constrained vertex/theta variables.
            for (size_t lji = 0; lji < 2; ++lji) {
                size_t ji = s.joint(lji);
                if (ji == NONE) continue;
                const auto &js = *jointSensitivity[lji];
                const size_t jo = m_dofOffsetForJoint[ji];
                if (mask.skipBRods && !js.is_A) {
                    v_local.template segment<3>(3 * (js.j + 1)).setZero();
                    v_local.template segment<3>(3 * (js.j    )).setZero();
                    v_local[local_theta_offset + js.j] = 0.0;
                    continue;
                }

                // Need to extract the perturbation for the correct length variable from the full perturbation vector.
                Eigen::Matrix<Real_, JointJacobianCols, 1> perturbation = v.template segment<JointJacobianCols>(jo + 3);
                perturbation[JointJacobianCols - 1] = v[jo + 3 + 4 + js.localSegmentIndex];

                Eigen::Matrix<Real_, JointJacobianRows, 1> delta_e_theta_p = js.jacobian * perturbation;

                v_local.template segment<3>(3 * (js.j + 1)) = v.template segment<3>(jo) + 0.5 * delta_e_theta_p.template segment<3>(0) + delta_e_theta_p.template segment<3>(4);
                v_local.template segment<3>(3 * (js.j    )) = v.template segment<3>(jo) - 0.5 * delta_e_theta_p.template segment<3>(0) + delta_e_theta_p.template segment<3>(4);
                v_local[local_theta_offset + js.j] = delta_e_theta_p[3];
            }
        }
        else { v_local.head(s.rod.numDoF()).setZero(); }

        if (variableDesignParameters) {
            if (mask.designParameter_in && m_umbrella_dPC.restLen && (s.segmentType() == SegmentType::Arm)) {
                const size_t local_dp_offset = s.rod.designParameterOffset();
                // Copy over the interior/free-end edge rest length perturbations.
                v_local.segment(local_dp_offset + s.hasStartJoint(), s.numFreeEdges())
                    = v.segment(m_restLenDofOffsetForSegment[si], s.numFreeEdges());

                // Copy constrained terminal edges' rest length perturbations from their controlling joint.
                for (size_t lji = 0; lji < 2; ++lji) {
                    size_t ji = s.joint(lji);
                    if (ji == NONE) continue;
                    const auto &js = *jointSensitivity[lji];
                    Real_ val = v[m_designParameterDoFOffsetForJoint[ji] + joint(ji).arm_offset_for_global_segment(si)];
                    if (mask.skipBRods && !js.is_A) val = 0.0;
                    v_local[local_dp_offset + js.j] = val;
                }
            }
            else { v_local.tail(s.rod.numDesignParameters()).setZero(); }
        }

        // Apply rod Hessian
        Hv_local.setZero(ndof_local);
        r.applyHessEnergy(v_local, Hv_local, variableDesignParameters, mask);

        // Apply dv_dr transpose (accumulate contribution to output gradient)
        if (mask.dof_out) {
            // Copy over the interior/free-end vertex and theta delta grad components
            const size_t free_vtx_components = 3 * s.numFreeVertices(),
                         local_theta_offset = s.rod.thetaOffset();
            Hv.segment(m_dofOffsetForSegment[si],                    free_vtx_components) = Hv_local.segment((3 * 2) * s.hasStartJoint(), free_vtx_components);
            Hv.segment(m_dofOffsetForSegment[si] + free_vtx_components, s.numFreeEdges()) = Hv_local.segment(local_theta_offset + s.hasStartJoint(), s.numFreeEdges());

            // Compute the perturbations of the constrained vertex/theta variables.
            for (size_t lji = 0; lji < 2; ++lji) {
                size_t ji = s.joint(lji);
                if (ji == NONE) continue;
                const auto &js = *jointSensitivity[lji];
                const size_t jo = m_dofOffsetForJoint[ji];
                if (mask.skipBRods && !js.is_A) continue;

                Eigen::Matrix<Real_, JointJacobianRows, 1> delta_grad_e_theta_p;
                delta_grad_e_theta_p << 0.5 * (Hv_local.template segment<3>(3 * (js.j + 1)) - Hv_local.template segment<3>(3 * js.j)),
                                       Hv_local[local_theta_offset + js.j],
                                       Hv_local.template segment<3>(3 * (js.j + 1)) + Hv_local.template segment<3>(3 * js.j);

                Hv.template segment<3>(jo    )        += Hv_local.template segment<3>(3 * (js.j + 1)) + Hv_local.template segment<3>(3 * js.j); // Joint position identity block
                Hv.template segment<4>(jo + 3)        += js.jacobian.template  leftCols<4>().transpose() * delta_grad_e_theta_p; // Joint orientation/angle Jacobian block
                Hv[jo + 3 + 4 + js.localSegmentIndex] += js.jacobian.template rightCols<1>().dot(delta_grad_e_theta_p);        // Joint length Jacobian block
            }
        }
        if (variableDesignParameters && mask.designParameter_out) {
            const size_t local_dp_offset = s.rod.designParameterOffset();
            if (m_umbrella_dPC.restLen && s.segmentType() == SegmentType::Arm) {
                // Copy over the interior/free-end edge rest length delta grad components.
                Hv.segment(m_restLenDofOffsetForSegment[si], s.numFreeEdges()) = Hv_local.segment(local_dp_offset + s.hasStartJoint(), s.numFreeEdges());

                // Accumulate constrained terminal edges' rest length delta grad components to their controlling joint.
                for (size_t lji = 0; lji < 2; ++lji) {
                    size_t ji = s.joint(lji);
                    if (ji == NONE) continue;
                    const auto &js = *jointSensitivity[lji];
                    if (mask.skipBRods && !js.is_A) continue;
                    Hv[m_designParameterDoFOffsetForJoint[ji] + joint(ji).arm_offset_for_global_segment(si)] += Hv_local[local_dp_offset + js.j];
                }
            }
        }

        // Compute joint Hessian term.
        if (mask.dof_in && mask.dof_out) {
            // typename ElasticRod_T<Real_>::Gradient sg(r);
            // sg.setZero();
            // Note: we only need the gradient with respect to the terminal
            // degrees of freedom, so we can ignore many of the energy contributions.
            const auto sg = r.template gradient<GradientStencilMaskTerminalsOnly>(updatedSource); // we never need the variable rest length gradient since the mapping from global to local rest lengths is linear

            // Accumulate contribution of the Hessian of e^j and theta^j wrt the joint parameters.
            //      dE/var^j (d^2 var^j / djoint_var_k djoint_var_l)
            for (size_t ji = 0; ji < 2; ++ji) {
                if (jointSensitivity[ji] == nullptr) continue;
                const auto &js = *jointSensitivity[ji];
                if (mask.skipBRods && !js.is_A) continue;
                const size_t o = segmentJointDofOffset[ji] + 3; // DoF index for first component of omega
                Eigen::Matrix<Real_, JointJacobianRows, 1> dE_djointvar;
                dE_djointvar << 0.5 * (sg.gradPos(js.j + 1) - sg.gradPos(js.j)),
                                sg.gradTheta(js.j),
                                sg.gradPos(js.j + 1) + sg.gradPos(js.j);
                Hv.template segment<4>(o)        += js.delta_jacobian.template  leftCols<4>().transpose() * dE_djointvar;
                Hv[o + 4 + js.localSegmentIndex] += js.delta_jacobian.template rightCols<1>().dot(dE_djointvar);
            }
        }
    };

    VecX result(VecX::Zero(v.size()));

    #if ENABLE_TBB
        RHALocalData<Real_> rhaLocalData;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numSegments()), make_rod_hessian_applier(applyPerSegmentHessian, v.size(), rhaLocalData));

        for (const auto &data : rhaLocalData)
            result += data.Hv;
    #else
        RodHessianApplierData<Real_> data;
        data.Hv.setZero(result.size());
        for (size_t si = 0; si < numSegments(); ++si)
            applyPerSegmentHessian(si, data);
        result = data.Hv;
    #endif

    return result;
}

template<typename Real_>
auto UmbrellaMesh_T<Real_>::applyHessianDeployment(const VecX &v, const HessianComputationMask &mask) const -> VecX {
    if (!(mask.dof_in && mask.dof_out)) return VecX::Zero(v.size()); // This term is dof-dof only...
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".applyHessianDeployment");
    VecX result = VecX::Zero(v.size());
    if (m_dftype == DeploymentForceType::LinearActuator) {
        m_linearActuator.addHessVec(*this, m_deploymentEnergyWeight, v, result);
        return result;
    }

    using M3d = Mat3_T<Real_>;
    VecX umbrellaHeights = getUmbrellaHeights();
    for (size_t ui = 0; ui < m_umbrella_to_top_bottom_joint_map.size(); ++ ui) {
        size_t    top_ji = m_umbrella_to_top_bottom_joint_map[ui][0];
        size_t bottom_ji = m_umbrella_to_top_bottom_joint_map[ui][1];
        size_t    top_ji_dof = dofOffsetForJoint(   top_ji);
        size_t bottom_ji_dof = dofOffsetForJoint(bottom_ji);
        Real_ height = umbrellaHeights[ui];
        if (height == 0) std::cout<<"Distance Deployment Hessian term encountered zero height umbrella at joint "<<ui<<"!"<<std::endl;
        
        Real_ coeff_one = m_deploymentEnergyWeight[ui] - m_deploymentEnergyWeight[ui] * m_targetDeploymentHeight[ui] / height;
        Real_ coeff_two = m_deploymentEnergyWeight[ui] - coeff_one;
        Vec3 tangent = (joint(top_ji).pos() - joint(bottom_ji).pos()) / height;
        M3d hessianBlock;
        if (m_dftype == DeploymentForceType::Constant) {
            hessianBlock = coeff_one * M3d::Identity() - (coeff_one * tangent) * tangent.transpose();
            if (height - m_targetDeploymentHeight[ui] != 0) hessianBlock /= abs(height - m_targetDeploymentHeight[ui]);
        }
        else hessianBlock = coeff_one * M3d::Identity() + (coeff_two * tangent) * tangent.transpose();
        
        result.template segment<3>(top_ji_dof) += hessianBlock * v.template segment<3>(top_ji_dof) - hessianBlock * v.template segment<3>(bottom_ji_dof);
        result.template segment<3>(bottom_ji_dof) += hessianBlock * v.template segment<3>(bottom_ji_dof) - hessianBlock * v.template segment<3>(top_ji_dof);
    }
    return result;
}

template<typename Real_>
auto  UmbrellaMesh_T<Real_>::applyHessianAttraction(const VecX &v, const HessianComputationMask &mask) const -> VecX {
    Real weight = (m_attraction_weight / (m_l0 * m_l0));
    if (!(mask.dof_in && mask.dof_out) || weight == 0) return VecX::Zero(v.size()); // This term is dof-dof only...
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".applyHessianAttraction");
    return weight * m_target_surface_fitter->applyHessian(*this, v);
}

template<typename Real_>
auto UmbrellaMesh_T<Real_>::applyHessian(const VecX &v, bool variableDesignParameters, const HessianComputationMask &mask, UmbrellaEnergyType type) const -> VecX {
    VecX result(VecX::Zero(v.size()));

    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::          Elastic) result += applyHessianElastic(v, variableDesignParameters, mask);
    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::       Deployment) result += applyHessianDeployment(v, mask);
    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::        Repulsion) { if (m_repulsionEnergyWeight != 0) throw std::runtime_error("Repulsion energy hessian vector product is not implemented!"); }
    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::       Attraction) result += applyHessianAttraction(v, mask);
    if (type == UmbrellaEnergyType::Full || type == UmbrellaEnergyType::AngleBoundPenalty) {
        if ((mask.dof_in && mask.dof_out) && (m_angleBoundEnforcement == AngleBoundEnforcement::Penalty)) {
            visitAngleBounds([&](size_t ji, Real_ lower, Real_ upper) {
                    size_t var = m_dofOffsetForJoint[ji] + 6;
                    result[var] += m_constraintBarrier.d2eval(joint(ji).alpha(), lower, upper) * v[var];
                });
        }
        
    }
    return result;
}

template<typename Real_>
auto UmbrellaMesh_T<Real_>::applyHessianPerArmRestlen(const VecX &v, const HessianComputationMask &mask, UmbrellaEnergyType type) const -> VecX {
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".applyHessianPARL");
    const size_t ndof = numExtendedDoFPARL();
    if (size_t(v.size()) != ndof) throw std::runtime_error("Input vector size mismatch");

    VecX vPerEdge(numExtendedDoF());
    size_t unchanged_length = numDoF();
    vPerEdge.head(unchanged_length) = v.head(unchanged_length);
    if (mask.designParameter_in && m_umbrella_dPC.restLen) m_armRestLenToEdgeRestLenMapTranspose.applyTransposeParallel(v.tail(numArmSegments()), vPerEdge.tail(numRestLengths()));
    else if (m_umbrella_dPC.restLen)                       vPerEdge.tail(numRestLengths()).setZero();
    auto HvPerEdge = applyHessian(vPerEdge, true, mask, type);

    VecX result(v.size());
    result.setZero();
    result.head(unchanged_length) = HvPerEdge.head(unchanged_length);
    if (mask.designParameter_out && m_umbrella_dPC.restLen) m_armRestLenToEdgeRestLenMapTranspose.applyRaw(HvPerEdge.tail(numRestLengths()).data(), result.tail(numArmSegments()).data(), /* no transpose */ false);
    else if (m_umbrella_dPC.restLen)                        result.tail(numArmSegments()).setZero();

    return result;
}

#endif /* end of include guard: UMBRELLAMESHHESSVEC_INL */

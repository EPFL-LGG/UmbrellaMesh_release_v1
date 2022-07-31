////////////////////////////////////////////////////////////////////////////////
// LinearActuator.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Models linear actuators that rigidly attach to the top and bottom plates
//  of umbrellas:
//      - the plates are both normal to the actuator axis
//        (we enforce agreement of the two plates' normals)
//      - the actuator passes through each plate center
//        (we apply a zero-restlength spring tangentially)
//      - the actuator enforces a given signed deployment height
//        (the height along the axis of the top plate is constrained to a
//        target value)
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef LINEARACTUATOR_HH
#define LINEARACTUATOR_HH

#include <Eigen/Dense>
#include <rotation_optimization.hh>
#include <MeshFEM/Parallelism.hh>

template<typename UM>
struct LinearActuators {
    using Real_  = typename UM::RealType;
    using V3d    = typename UM::Vec3;
    using M3d    = typename UM::Mat3;
    using VXd    = typename UM::VecX;
    using Joint  = typename UM::Joint;
    using CSCMat = typename UM::CSCMat;
    using ropt   = rotation_optimization<Real_>;

    // Gradients of a single actuator's energy with respect to its involved variables
    // (u, ntop, nbot)
    // where u = p_top - p_bot
    using ElementGradient = Eigen::Matrix<Real_, 9, 1>;
    using ElementHessian  = Eigen::Matrix<Real_, 9, 9>;

    LinearActuators() {}

    using SerializedState = std::tuple<Real_, Real_, Real_>;
    SerializedState getState() const {
        return std::make_tuple(angleStiffness, tangentialStiffness, axialStiffness);
    }
    void setState(const SerializedState &state) {
        angleStiffness      = std::get<0>(state);
        tangentialStiffness = std::get<1>(state);
        axialStiffness      = std::get<2>(state);
    }

    LinearActuators(const SerializedState &state) { setState(state); }

    template<typename UM2>
    LinearActuators(const LinearActuators<UM2> &la) { setState(la.getState()); }

    // Energy of a single actuator
    Real_ energy(const Joint &top, const Joint &bot, Real_ h_tgt) const {
        const V3d &ntop = top.normal(),
                  &nbot = bot.normal();
        Real_ E = angleStiffness * ntop.dot(nbot);       // alignment: 0.5 * ||n1 + n2||^2 (note ||n1|| = ||n2|| = 1 = const)
        V3d   a = (ntop - nbot).normalized();
        V3d   u = top.pos() - bot.pos();
        Real_ h = a.dot(u);
        E += 0.5 * axialStiffness * (h - h_tgt) * (h - h_tgt);      // height enforcement
        E += 0.5 * tangentialStiffness * (u.squaredNorm() - h * h); // zero-restlength tangential spring
        return E;
    }

    Real_ energy(const UM &um, const VXd &weight) const {
        Real_ result = 0;
        VXd h_tgt = um.getTargetDeploymentHeightVector();
        const size_t nu = um.numUmbrellas();
        for (size_t ui = 0; ui < nu; ++ui) {
            result += weight[ui] * energy(um.joint(um.getUmbrellaCenterJi(ui, 0)),
                                          um.joint(um.getUmbrellaCenterJi(ui, 1)), h_tgt[ui]);
        }
        return result;
    }

    // Gradient of a single actuator's energy with respect to the involved u vector and normals
    ElementGradient grad_un(const Joint &top, const Joint &bot, Real_ h_tgt) const {
        const V3d &ntop = top.normal(),
                  &nbot = bot.normal();
        V3d a_hat = ntop - nbot;
        Real_ anorm = a_hat.norm();
        a_hat /= anorm;
        V3d   u = top.pos() - bot.pos();
        Real_ h = a_hat.dot(u);

        Real_ dE_dh = -tangentialStiffness * h + axialStiffness * (h - h_tgt); // dE/dh

        ElementGradient result;
        auto dE_du    = result.template segment<3>(0),
             dE_dntop = result.template segment<3>(3),
             dE_dnbot = result.template segment<3>(6);

        // tangential + axial terms
        dE_dntop = (u - h * a_hat) * (dE_dh / anorm);
        dE_dnbot = -dE_dntop;

        // alignment term
        dE_dntop += angleStiffness * nbot;
        dE_dnbot += angleStiffness * ntop;

        dE_du = dE_dh * a_hat + tangentialStiffness * u;
        return result;
    }

    // (Upper triangle of) Hessian of a single actuator's energy with respect
    // to the involved u vector and normals.
    ElementHessian hess_un(const Joint &top, const Joint &bot, Real_ h_tgt) const {
        const V3d &ntop = top.normal(),
                  &nbot = bot.normal();
        V3d a_hat   = ntop - nbot;
        Real_ anorm = a_hat.norm();
        a_hat /= anorm;
        V3d u = top.pos() - bot.pos();
        Real_ h = a_hat.dot(u);
        V3d dh_da = (u - u.dot(a_hat) * a_hat) / anorm;

        Real_ dE_dh   = -tangentialStiffness * h + axialStiffness * (h - h_tgt); // dE/dh
        Real_ d2E_dh2 = -tangentialStiffness + axialStiffness; // d^22E/dh^2

        ElementHessian result;

        M3d d2E_da2  = d2E_dh2 * (dh_da * dh_da.transpose()) - (dE_dh / anorm) * (h / anorm * (M3d::Identity() - a_hat * a_hat.transpose()) + a_hat * dh_da.transpose() + dh_da * a_hat.transpose());
        M3d d2E_duda = d2E_dh2 * (a_hat * dh_da.transpose()) + (dE_dh / anorm) *              (M3d::Identity() - a_hat * a_hat.transpose());

        result.template block<3, 3>(0, 0) = tangentialStiffness * M3d::Identity() + d2E_dh2 * (a_hat * a_hat.transpose()); // d^2E/du^2
        result.template block<3, 3>(0, 3) =  d2E_duda;                                   // d^2E/dudn1
        result.template block<3, 3>(0, 6) = -d2E_duda;                                   // d^2E/dudn2
        result.template block<3, 3>(3, 3) =  d2E_da2;                                    // d^2E/dn1dn1
        result.template block<3, 3>(3, 6) = -d2E_da2 + angleStiffness * M3d::Identity(); // d^2E/dn1dn2
        result.template block<3, 3>(6, 6) =  d2E_da2;                                    // d^2E/dn2dn2

        return result;
    }

    void addGradient(const UM &um, const VXd &weight, VXd &g) const {
        const size_t nu = um.numUmbrellas();
        VXd h_tgt = um.getTargetDeploymentHeightVector();
        for (size_t ui = 0; ui < nu; ++ui) {
            int jitop = um.getUmbrellaCenterJi(ui, 0),
                jibot = um.getUmbrellaCenterJi(ui, 1);
            auto &jtop = um.joint(jitop);
            auto &jbot = um.joint(jibot);
            ElementGradient local_grad = weight[ui] * grad_un(jtop, jbot, h_tgt[ui]);

            g.template segment<3>(um.dofOffsetForJoint(jitop) + 0) += local_grad.template segment<3>(0); // p_top
            g.template segment<3>(um.dofOffsetForJoint(jibot) + 0) -= local_grad.template segment<3>(0); // p_bot
            g.template segment<3>(um.dofOffsetForJoint(jitop) + 3) += (ropt::grad_rotated_vector(jtop.omega(), jtop.source_normal())).transpose() * local_grad.template segment<3>(3); // omega_top
            g.template segment<3>(um.dofOffsetForJoint(jibot) + 3) += (ropt::grad_rotated_vector(jbot.omega(), jbot.source_normal())).transpose() * local_grad.template segment<3>(6); // omega_bot
        }
    }

    template<class SPMat>
    void addHessian(const UM &um, const VXd &weight, SPMat &H) const {
        const size_t nu = um.numUmbrellas();
        VXd h_tgt = um.getTargetDeploymentHeightVector();
        for (size_t ui = 0; ui < nu; ++ui) {
            int jitop = um.getUmbrellaCenterJi(ui, 0),
                jibot = um.getUmbrellaCenterJi(ui, 1);
            auto &jtop = um.joint(jitop);
            auto &jbot = um.joint(jibot);
            ElementGradient local_grad = weight[ui] * grad_un(jtop, jbot, h_tgt[ui]);
            ElementHessian  local_hess = weight[ui] * hess_un(jtop, jbot, h_tgt[ui]);

            size_t jo_top = um.dofOffsetForJoint(jitop);
            size_t jo_bot = um.dofOffsetForJoint(jibot);
            size_t jo_min = std::min(jo_top, jo_bot);
            size_t jo_max = std::max(jo_top, jo_bot);

            H.addNZBlock(jo_top, jo_top,  local_hess.template block<3, 3>(0, 0)); // (p_top, p_top)
            H.addNZBlock(jo_bot, jo_bot,  local_hess.template block<3, 3>(0, 0)); // (p_bot, p_bot)
            H.addNZBlock(jo_min, jo_max, -local_hess.template block<3, 3>(0, 0)); // (p_top, p_bot) or (p_bot, p_top)

            M3d dntop_domega = ropt::grad_rotated_vector(jtop.omega(), jtop.source_normal());
            M3d dnbot_domega = ropt::grad_rotated_vector(jbot.omega(), jbot.source_normal());
            M3d dE_dudntop = local_hess.template block<3, 3>(0, 3) * dntop_domega;
            M3d dE_dudnbot = local_hess.template block<3, 3>(0, 6) * dnbot_domega;

            H.addNZBlock(jo_top, jo_top + 3,  dE_dudntop); // (p_top, n_top)
            H.addNZBlock(jo_bot, jo_bot + 3, -dE_dudnbot); // (p_bot, n_bot)
            if (jo_bot < jo_top) { H.addNZBlock(jo_bot, jo_top + 3, -dE_dudntop); H.addNZBlock(jo_bot + 3, jo_top,  dE_dudnbot.transpose()); } // (p_bot, n_top), (n_bot, p_top)
            else                 { H.addNZBlock(jo_top, jo_bot + 3,  dE_dudnbot); H.addNZBlock(jo_top + 3, jo_bot, -dE_dudntop.transpose()); } // (p_top, n_bot), (n_top, p_bot)

            H.addNZBlock(jo_top + 3, jo_top + 3, dntop_domega.transpose() * local_hess.template block<3, 3>(3, 3) * dntop_domega + ropt::d_contract_hess_rotated_vector(jtop.omega(), jtop.source_normal(), /* dE/dn_top */ local_grad.template segment<3>(3))); // (n_top, n_top)
            H.addNZBlock(jo_bot + 3, jo_bot + 3, dnbot_domega.transpose() * local_hess.template block<3, 3>(6, 6) * dnbot_domega + ropt::d_contract_hess_rotated_vector(jbot.omega(), jbot.source_normal(), /* dE/dn_bot */ local_grad.template segment<3>(6))); // (n_bot, n_bot)

            if (jo_bot < jo_top) H.addNZBlock(jo_bot + 3, jo_top + 3, dnbot_domega.transpose() * local_hess.template block<3, 3>(3, 6) * dntop_domega); // (n_bot, n_top)
            else                 H.addNZBlock(jo_top + 3, jo_bot + 3, dntop_domega.transpose() * local_hess.template block<3, 3>(3, 6) * dnbot_domega); // (n_top, n_bot)
        }
    }

    void addHessVec(const UM &um, const VXd &weight, Eigen::Ref<const VXd> v, Eigen::Ref<VXd> delta_g) const {
        VXd h_tgt = um.getTargetDeploymentHeightVector();
        // Note: all umbrellas are independent, so we can do this in parallel lock-free
        parallel_for_range(um.numUmbrellas(), [&](size_t ui) {
            int jitop = um.getUmbrellaCenterJi(ui, 0),
                jibot = um.getUmbrellaCenterJi(ui, 1);
            auto &jtop = um.joint(jitop);
            auto &jbot = um.joint(jibot);
            ElementGradient local_grad = weight[ui] * grad_un(jtop, jbot, h_tgt[ui]);
            ElementHessian  local_hess = weight[ui] * hess_un(jtop, jbot, h_tgt[ui]);
            size_t jo_top = um.dofOffsetForJoint(jitop);
            size_t jo_bot = um.dofOffsetForJoint(jibot);

            M3d dntop_domega = ropt::grad_rotated_vector(jtop.omega(), jtop.source_normal());
            M3d dnbot_domega = ropt::grad_rotated_vector(jbot.omega(), jbot.source_normal());

            // pos-pos
            V3d delta_u = (v.template segment<3>(jo_top) - v.template segment<3>(jo_bot));
            V3d delta_dE_du = local_hess.template block<3, 3>(0, 0) * delta_u;

            // pos-omega
            M3d dE_dudntop = local_hess.template block<3, 3>(0, 3) * dntop_domega;
            M3d dE_dudnbot = local_hess.template block<3, 3>(0, 6) * dnbot_domega;
            delta_dE_du += dE_dudntop * v.template segment<3>(jo_top + 3)
                         + dE_dudnbot * v.template segment<3>(jo_bot + 3);

            // omega-pos
            V3d delta_dE_dntop = dE_dudntop.transpose() * delta_u;
            V3d delta_dE_dnbot = dE_dudnbot.transpose() * delta_u;

            // omega-omega
            delta_dE_dntop += (dntop_domega.transpose() * local_hess.template block<3, 3>(3, 3) * dntop_domega + ropt::d_contract_hess_rotated_vector(jtop.omega(), jtop.source_normal(), /* dE/dn_top */ local_grad.template segment<3>(3))) * v.template segment<3>(jo_top + 3) + (dntop_domega.transpose() * local_hess.template block<3, 3>(3, 6) * dnbot_domega) * v.template segment<3>(jo_bot + 3);
            delta_dE_dnbot += (dnbot_domega.transpose() * local_hess.template block<3, 3>(6, 6) * dnbot_domega + ropt::d_contract_hess_rotated_vector(jbot.omega(), jbot.source_normal(), /* dE/dn_bot */ local_grad.template segment<3>(6))) * v.template segment<3>(jo_bot + 3) + (dnbot_domega.transpose() * local_hess.template block<3, 3>(3, 6) * dntop_domega) * v.template segment<3>(jo_top + 3);

            delta_g.template segment<3>(jo_top)     += delta_dE_du;
            delta_g.template segment<3>(jo_bot)     -= delta_dE_du;
            delta_g.template segment<3>(jo_top + 3) += delta_dE_dntop;
            delta_g.template segment<3>(jo_bot + 3) += delta_dE_dnbot;
        });
    }

    // Statistics
    VXd plateHeights(const UM &um) const {
        const size_t nu = um.numUmbrellas();
        VXd result(nu);
        for (size_t ui = 0; ui < nu; ++ui) {
            auto &top = um.joint(um.getUmbrellaCenterJi(ui, 0));
            auto &bot = um.joint(um.getUmbrellaCenterJi(ui, 1));
            V3d a = (top.normal() - bot.normal()).normalized();
            V3d u = top.pos() - bot.pos();
            result[ui] = a.dot(u);
        }
        return result;
    }

    // Stiffness of individual constituent springs (these are ultimately
    // relative to the overall deployment weight)
    Real_ angleStiffness      = 1.0;
    Real_ tangentialStiffness = 1.0;
    Real_ axialStiffness      = 1.0;
};

#endif /* end of include guard: LINEARACTUATOR_HH */

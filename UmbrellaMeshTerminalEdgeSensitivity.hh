#ifndef UMBRELLAMESHTERMINALEDGESENSITIVITY_HH
#define UMBRELLAMESHTERMINALEDGESENSITIVITY_HH

#include "UmbrellaMesh.hh"

template<typename Derived, typename Real_> std::enable_if_t<Derived::RowsAtCompileTime == 0> unpack_delta_jparams(const Eigen::MatrixBase<Derived> & , const size_t           , Eigen::Matrix<Real_, 3, 1>,              Real_ &,            Real_ &         ) { throw std::logic_error("Fail."); }
template<typename Derived, typename Real_> std::enable_if_t<Derived::RowsAtCompileTime == 5> unpack_delta_jparams(const Eigen::MatrixBase<Derived> &v, const size_t len_offset, Eigen::Matrix<Real_, 3, 1> &delta_omega, Real_ &delta_alpha, Real_ &delta_len) { delta_omega = v.head(3); delta_alpha = v[3]; delta_len = v[len_offset];  }

// Store the derivatives of the constrained terminal edges' edge vectors
// and material frame angles with respect to the controlling joint's 
// parameters. This is represented by the following Jacobian matrix:
//  e^j     [     de^j/d_omega        de^j/d_alpha    de^j/d_len_X ]
//  theta^j [ dtheta^j/d_omega    dtheta^j/d_alpha                0]
//  p^j     [     dp^j/d_omega        dp^j/d_alpha                0]
//
// To obtain the change in the centerline positions, this Jacobian
// should be composed with the following linear map:
//           pos     e_X     theta^j  p_X
// x_j     [  I    - 0.5 I     0       I] [ I 0 ... 0]
// x_{j+1} [  I      0.5 I     0       I] [ jacobian ]
// theta^j [  0          0     I       0]
// (here, "x_j" and "x_{j+1}" are the tail and tip vertex centerline positions)
template<typename Real_>
struct UmbrellaMeshTerminalEdgeSensitivity {
    using Vec3 = Vec3_T<Real_>;
    using Mat3 = Mat3_T<Real_>;
    using Joint = typename UmbrellaMesh_T<Real_>::Joint;
    using ropt  = typename UmbrellaMesh_T<Real_>::ropt;

    // Whether edge j is part of the joint's rod "A" or "B",
    // and whether its orientation agrees with e_A/e_B.
    size_t j;
    bool is_A;
    size_t localSegmentIndex;

    static constexpr size_t JacobianRows = 7;
    static constexpr size_t JacobianCols = 5;

    // Derivative of [e^j, theta^j, p^j]^T with respect to [omega, alpha, len_X]
    Eigen::Matrix<Real_, JacobianRows, JacobianCols> jacobian;

    // hessian[i] holds the Hessian of the i^th component of [e^j, theta^j, p^j]^T
    // with respect to (omega, alpha, len_X).
    std::array<Eigen::Matrix<Real_, JacobianCols, JacobianCols>, JacobianRows> hessian;

    // Directional derivative of jacobian (evaluated if update was called with a joint parameter perturbation vector).
    Eigen::Matrix<Real_, JacobianRows, JacobianCols> delta_jacobian;

    // Leave uninitialized; update must be called before this instance is used!
    UmbrellaMeshTerminalEdgeSensitivity() { }

    UmbrellaMeshTerminalEdgeSensitivity(const Joint &joint, size_t si, const ElasticRod_T<Real_> &rod, bool updatedSource, bool evalHessian = false) {
        update(joint, si, rod, updatedSource, evalHessian);
    }

    void update(const Joint &joint, size_t si, const ElasticRod_T<Real_> &rod, bool updatedSource, bool evalHessian) {
        update(joint, si, rod, updatedSource, evalHessian, Eigen::Matrix<Real, 0, 0>());
    }

    template<typename Derived>
    void update(const Joint &joint, size_t si, const ElasticRod_T<Real_> &rod, bool updatedSource, bool evalHessian, const Eigen::MatrixBase<Derived> &delta_jparams) {
        bool isStart;
        std::tie(is_A, isStart, localSegmentIndex) = joint.terminalEdgeIdentification(si);

        j = isStart ? 0 : rod.numEdges() - 1;

        const auto &dc = rod.deformedConfiguration();

        const auto &t  = dc.tangent[j],
                   &d1 = dc.materialFrame[j].d1,
                   // &mp = (dc.point(j) + dc.point(j + (isStart ? 1 : -1))) / 2 - joint.pos(),
                   &w  = joint.omega();
        const auto &ns = joint.source_normal(localSegmentIndex),
                   &n  = joint.normal(localSegmentIndex);
                  
        const auto  tsX = joint.source_t(localSegmentIndex);
        const auto  psX = joint.source_p(localSegmentIndex);
        const Real_ len = joint.len(localSegmentIndex);
        const size_t len_offset = 4,
                   alpha_offset = 3,
                   theta_offset = 3,
                   midpoint_offset = 4;

        const auto  tiX = joint.input_t(localSegmentIndex).normalized();
        const auto  piX = joint.input_p(localSegmentIndex);
        const auto  niX = joint.input_normal(localSegmentIndex).normalized();

        // The following code has inlined, optimized expressions for the quantities:
        //      Vec3 dt_dalpha   = ropt::rotated_vector(w, ns.cross(tsX));
        //      Vec3 tX          = ropt::rotated_vector(w, tsX);
        //      Mat3 dtX_domega  = ropt::grad_rotated_vector(w, tsX);
        //      Mat3 d_n_d_omega = ropt::grad_rotated_vector(w, ns); // actually we compute -dc.materialFrame[j].d1 dotted with this quantity
        // These quantities share intermediate values and are a bottleneck, especially for autodiff types
        const Real_ theta_sq    = w.squaredNorm();

        // Use simpler formulas for rotation variations around the identity
        // (But only if we're using a native floating point type; for autodiff
        // types, we need the full formulas).
        const bool variation_around_identity = (theta_sq == 0) && (std::is_arithmetic<Real_>::value);

        const Real_ theta       = sqrt(theta_sq);
        const Real_ sinc_th     = sinc(theta, theta_sq),
                    omcdthsq    = one_minus_cos_div_theta_sq(theta, theta_sq),
                    tcmsdtc     = theta_cos_minus_sin_div_theta_cubed(theta, theta_sq),
                    tcm2ptsdtp4 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq);
        
        auto get_domega_v_d_omega = [&](const Vec3 v) {
            const Vec3 neg_v_sinc = v * (-sinc_th);
            const Real_ w_dot_v   = w.dot(v);
            const Vec3 w_cross_v  = w.cross(v);
            Mat3 domega_v_domega;
            domega_v_domega <<       0, -neg_v_sinc[2],  neg_v_sinc[1],
                  neg_v_sinc[2],                0, -neg_v_sinc[0],
                 -neg_v_sinc[1],  neg_v_sinc[0],                0;
            if (!variation_around_identity) {
                domega_v_domega += (neg_v_sinc + w_cross_v * tcmsdtc + w * (w_dot_v * tcm2ptsdtp4)) * w.transpose() + (omcdthsq * w) * v.transpose();
                domega_v_domega.diagonal().array() += w_dot_v * omcdthsq;
            }
            return domega_v_domega;
        };

        // Compute terms needed for d eX / d w.
        const Real_ w_dot_tsX  = w.dot(tsX);
        const Vec3 w_cross_tsX = w.cross(tsX);
        const Vec3 tX          = t;

        Mat3 sensitivity_frame = (is_A ? joint.get_ghost_alpha_sensitivity_frame_A() : joint.get_ghost_alpha_sensitivity_frame_B());
        Vec3 sensitivity_tiX   = sensitivity_frame * tiX;
        const Vec3 dt_dalpha   = ropt::rotated_vector(w, sensitivity_tiX);

        Mat3 dtX_domega = get_domega_v_d_omega(tsX);
        ///////////////////////////////////////

        // Compute terms needed for d pX / d w.
        const Real_ w_dot_psX   = w.dot(psX);
        const Vec3 w_cross_psX  = w.cross(psX);

        Vec3 sensitivity_piX  = sensitivity_frame * piX;
        const Vec3 dmp_dalpha = ropt::rotated_vector(w, sensitivity_piX);

        Mat3 dpX_domega = get_domega_v_d_omega(psX);
    
        Vec3 sensitivity_niX = sensitivity_frame * niX;
        const Vec3 dn_dalpha = ropt::rotated_vector(w, sensitivity_niX);

        Mat3 d_n_d_omega = get_domega_v_d_omega(ns);
        ///////////////////////////////////////

        const Vec3 neg_ns_sinc = ns * (-sinc_th); //Only used in expressions where ns is assumed to be the rod's source normal and not the joint's
        Vec3 w_cross_ns, d_n_d_omega_w_coeff;
        Real_ w_dot_ns = 0;
        if (!variation_around_identity) {
            w_cross_ns = w.cross(ns);
            w_dot_ns = w.dot(ns);
            d_n_d_omega_w_coeff = (neg_ns_sinc + w_cross_ns * tcmsdtc + w * (w_dot_ns * tcm2ptsdtp4));
        }

        jacobian.setZero();
        jacobian.template block<3, 3>(0,            0) = dtX_domega * len;
        jacobian.template block<3, 1>(0, alpha_offset) = dt_dalpha * len; // d e^j / d_alpha
        jacobian.template block<3, 1>(0,   len_offset) = tX;

        // Gradient due to the rotating joint normal
        // jacobian.template block<1, 3>(theta_offset, 0) = -d1.transpose() * d_n_d_omega;
        if (variation_around_identity) jacobian.template block<1, 3>(theta_offset, 0) = neg_ns_sinc.cross(d1).transpose();
        else                           jacobian.template block<1, 3>(theta_offset, 0) = (neg_ns_sinc.cross(d1).transpose() - (d1.dot(d_n_d_omega_w_coeff)) * w.transpose() - ((d1.dot(w)) * omcdthsq) * ns.transpose() - ( w_dot_ns * omcdthsq) * d1.transpose());
        jacobian(theta_offset, alpha_offset) = -d1.dot(dn_dalpha);


        jacobian.template block<3, 3>(midpoint_offset,            0) = dpX_domega;
        jacobian.template block<3, 1>(midpoint_offset, alpha_offset) = dmp_dalpha;

        // Contribution due to the rotating reference directors.
        // (Zero if the source frame is updated: if we're evaluating at the
        // source frame configuration, the reference directors don't rotate
        // around the tangent.)
        if (!updatedSource) {
            const auto &rds1 = dc.sourceReferenceDirectors[j].d1,
                       & rd1 = dc.referenceDirectors[j].d1,
                       & rd2 = dc.referenceDirectors[j].d2,
                       &  ts = dc.sourceTangent[j];
            // As t approaches -ts, the parallel transport operator becomes singular;
            // see discussion in the parallelTransportNormalized function in VectorOperations.hh.
            // To avoid NaNs in this case, we arbitrarily define the parallel
            // transport operator to be the identity (so its derivative is 0).
            const Real_ chi_hat = 1.0 + ts.dot(t);
            if (std::abs(stripAutoDiff(chi_hat)) > 1e-14) {
                Real_ inv_chi_hat = 1.0 / chi_hat;
                Real_ rds1_dot_t = rds1.dot(t),
                      rd2_dot_ts = rd2.dot(ts);

                // Angular velocity of the reference director as the terminal edge is perturbed
                Vec3 neg_d_theta_ref_dt = inv_chi_hat * (rds1_dot_t * rd1.cross(ts) + rd2_dot_ts * rds1)
                                        - (inv_chi_hat * inv_chi_hat * rds1_dot_t * rd2_dot_ts) * ts
                                        + rds1.cross(rd1);
                jacobian.template block<1, 3>(theta_offset, 0) += neg_d_theta_ref_dt.transpose() * dtX_domega;
                jacobian(theta_offset, alpha_offset) += neg_d_theta_ref_dt.dot(dt_dalpha);
            }
        }



        if (!evalHessian) return;

        const Real_ eptsmecmftsdtp6 = eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(theta, theta_sq);
        const Real_ ttcptsm3sdtp5   = three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5(theta, theta_sq);
        const Vec3 w_tcm2ptsdtp4 = tcm2ptsdtp4 * w;

    
        // dtperp_domega = (n x dtX_domega) - (tX x d_n_d_omega) = (n x dt_dalpha) * dt_dalpha.transpose() * dtX_domega - (tX x dt_dalpha) * (dt_dalpha.transpose() * dtX_domega)
        // TODO: not used in hessian.
        // const auto tperp_dot_dtX_domega = (d1.transpose() * dtX_domega).eval();
        // TODO: Need to reconsider this formula: const Mat3 dtperp_domega = -tX * tperp_dot_dtX_domega - n * (dt_dalpha.transpose() * d_n_d_omega);

        constexpr size_t delta_size = Derived::RowsAtCompileTime;
        static_assert((delta_size == 0) || (delta_size == 5), "Invalid delta joint parameter vector size");

        // d2v / d alpha d alpha 
        Mat3 hessian_frame = (is_A ? joint.get_ghost_alpha_hessian_frame_A() : joint.get_ghost_alpha_hessian_frame_B());
        Vec3 hessian_tiX = hessian_frame * tiX;
        Vec3 hessian_piX = hessian_frame * piX;
        Vec3 hessian_niX = hessian_frame * niX;
        const Vec3 d2t_dalpha2  = ropt::rotated_vector(w, hessian_tiX);
        const Vec3 d2mp_dalpha2 = ropt::rotated_vector(w, hessian_piX);
        const Vec3 d2n_dalpha2  = ropt::rotated_vector(w, hessian_niX);

        // Evaluate full Hessian
        Mat3 d_t_d_alpha_d_omega = get_domega_v_d_omega(sensitivity_tiX); 
        Mat3 d_p_d_alpha_d_omega = get_domega_v_d_omega(sensitivity_piX);

        ////////////////////////////////////////////////////////////////////
        // Prepare Theta hessian
        ////////////////////////////////////////////////////////////////////

        Mat3 d1_dot_pder2_n_domega_domega_presym;
        {
            if (variation_around_identity) {
                const Vec3 half_ns = 0.5 * ns;
                d1_dot_pder2_n_domega_domega_presym = d1 * half_ns.transpose();
                // d1_dot_pder2_n_domega_domega_presym.diagonal().array() += -d1.dot(half_ns); // Note: d1 should be perpendicular to ns in this case!
            }
            else {
                const Real_ w_dot_d1 = w.dot(d1);
                d1_dot_pder2_n_domega_domega_presym  =  d1 * (omcdthsq * ns + w_dot_ns * w_tcm2ptsdtp4).transpose()
                          + ((0.5 *  (w_dot_ns * eptsmecmftsdtp6 * w_dot_d1 - ttcptsm3sdtp5 * w_cross_ns.dot(d1) - tcmsdtc * ns.dot(d1))) * w +  tcmsdtc * ns.cross(d1) + ( tcm2ptsdtp4 * w_dot_d1) * ns) * w.transpose();
                d1_dot_pder2_n_domega_domega_presym.diagonal().array() += ( 0.5) * d1.dot(d_n_d_omega_w_coeff);
            }
        }

        Mat3 d_d1_d_omega;
        d_d1_d_omega.col(0) = d_n_d_omega.col(0).cross(t) + n.cross(dtX_domega.col(0));
        d_d1_d_omega.col(1) = d_n_d_omega.col(1).cross(t) + n.cross(dtX_domega.col(1));
        d_d1_d_omega.col(2) = d_n_d_omega.col(2).cross(t) + n.cross(dtX_domega.col(2));

        // const Mat3 presym_block = (-0.5) * (d_n_d_omega.transpose() * dtperp_domega) - d1_dot_pder2_n_domega_domega_presym;
        const Mat3 presym_block = - 0.5 * (d_n_d_omega.transpose() * d_d1_d_omega) - d1_dot_pder2_n_domega_domega_presym;

        // dn_dalpha = R(w) sensitivity_niX
        // dn^2_d_omega_d_alpha = d(R(w) sensitivity_niX) dw
        Mat3 d_n_d_alpha_d_omega = get_domega_v_d_omega(sensitivity_niX);
        
        Vec3 minus_dn_dalpha_dot_d_d1_domega = - dn_dalpha.transpose() * d_d1_d_omega;
        Vec3 minus_d1_dot_pder2_n_domega_dalpha = -d1.transpose() * d_n_d_alpha_d_omega;
        Vec3 minus_sign_dt_domega_dot_t_cross_dt_dalpha = -0.5 * dtX_domega.transpose() * (tX.cross(dt_dalpha));
        const Vec3 d2_theta_dalpha_domega =  minus_dn_dalpha_dot_d_d1_domega + minus_d1_dot_pder2_n_domega_dalpha + minus_sign_dt_domega_dot_t_cross_dt_dalpha;
        Vec3 d_d1_d_alpha = dn_dalpha.cross(t) + n.cross(dt_dalpha);

        if (delta_size == 0) {
            for (size_t i = 0; i < JacobianRows; ++i) { hessian[i].setZero(); }

            ///////////////////////////////////////////////////////////////////
            // e^j hessian
            ///////////////////////////////////////////////////////////////////
            // d^2 e^j / d_omega d_omega
            {
                const Vec3 v = len * tsX;
                if (variation_around_identity) {
                    const Vec3 half_v = 0.5 * v;
                    for (size_t i = 0; i < 3; ++i) {
                        // hess_comp[i] = -v[i] * I + 0.5 * (I.col(i) * v.transpose() + v * I.row(i));
                        auto dst = hessian[i].template block<3, 3>(0, 0);
                        dst.diagonal().array() = -v[i];
                        dst.row(i) += half_v.transpose();
                        dst.col(i) += half_v;
                    }
                }
                else {
                    const Real_ w_dot_v = len * w_dot_tsX;
                    const Mat3 v_cross_term = ropt::cross_product_matrix(tcmsdtc * v);
                    const Vec3 v_cross_w = (-len) * w_cross_tsX;
                    for (size_t i = 0; i < 3; ++i) {
                        auto dst = hessian[i].template block<3, 3>(0, 0);
                        dst = ((0.5 * (w_dot_v * eptsmecmftsdtp6 * w[i] + ttcptsm3sdtp5 * v_cross_w[i] - tcmsdtc * v[i])) * w + v_cross_term.col(i) + w_tcm2ptsdtp4[i] * v) * w.transpose();
                        dst.col(i) += omcdthsq * v + w_dot_v * w_tcm2ptsdtp4;
                        dst += dst.transpose().eval();
                        dst.diagonal().array() += w_dot_v * w_tcm2ptsdtp4[i] - sinc_th * v[i] - tcmsdtc * v_cross_w[i];
                    }
                }
            }

            // d^2 e^j / d_omega d_len
            for (size_t i = 0; i < 3; ++i) {
                hessian[i].template block<3, 1>(0, len_offset) = dtX_domega.row(i).transpose();
                hessian[i].template block<1, 3>(len_offset, 0) = dtX_domega.row(i);
            }

            // d^2 e^j / d_omega d_alpha, d^2 e^j / d_alpha d_len, d^2 e^j / d_alpha d_alpha
            for (size_t i = 0; i < 3; ++i) {
                hessian[i].template block<3, 1>(0, alpha_offset) = d_t_d_alpha_d_omega.row(i).transpose() * len; // (omega, alpha)
                hessian[i].template block<1, 3>(alpha_offset, 0) = d_t_d_alpha_d_omega.row(i) * len;             // (alpha, omega)
                hessian[i](len_offset,   alpha_offset) = dt_dalpha[i]; // (alpha,   len)
                hessian[i](alpha_offset,   len_offset) = dt_dalpha[i]; // (  len, alpha)
                hessian[i](alpha_offset, alpha_offset) = d2t_dalpha2[i] * len; // (alpha, alpha)
            }
            ///////////////////////////////////////////////////////////////////
            // p^j hessian
            ///////////////////////////////////////////////////////////////////
            // d^2 p^j / d_omega d_omega
            {

                const Vec3 v = psX;
                if (variation_around_identity) {
                    const Vec3 half_v = 0.5 * v;
                    for (size_t i = 0; i < 3; ++i) {
                        // hess_comp[i] = -v[i] * I + 0.5 * (I.col(i) * v.transpose() + v * I.row(i));
                        auto dst = hessian[i + midpoint_offset].template block<3, 3>(0, 0);
                        dst.diagonal().array() = -v[i];
                        dst.row(i) += half_v.transpose();
                        dst.col(i) += half_v;
                    }
                }
                else {
                    const Real_ w_dot_v = w_dot_psX;
                    const Mat3 v_cross_term = ropt::cross_product_matrix(tcmsdtc * v);
                    const Vec3 v_cross_w = -w_cross_psX;
                    for (size_t i = 0; i < 3; ++i) {
                        auto dst = hessian[i + midpoint_offset].template block<3, 3>(0, 0);
                        dst = ((0.5 * (w_dot_v * eptsmecmftsdtp6 * w[i] + ttcptsm3sdtp5 * v_cross_w[i] - tcmsdtc * v[i])) * w + v_cross_term.col(i) + w_tcm2ptsdtp4[i] * v) * w.transpose();
                        dst.col(i) += omcdthsq * v + w_dot_v * w_tcm2ptsdtp4;
                        dst += dst.transpose().eval();
                        dst.diagonal().array() += w_dot_v * w_tcm2ptsdtp4[i] - sinc_th * v[i] - tcmsdtc * v_cross_w[i];
                    }
                }
            }

            // d^2 p^j / d_omega d_alpha, d^2 p^j / d_alpha d_alpha
            for (size_t i = 0; i < 3; ++i) {
                hessian[i + midpoint_offset].template block<3, 1>(0, alpha_offset) = d_p_d_alpha_d_omega.row(i).transpose(); // (omega, alpha)
                hessian[i + midpoint_offset].template block<1, 3>(alpha_offset, 0) = d_p_d_alpha_d_omega.row(i);             // (alpha, omega)
                hessian[i + midpoint_offset](alpha_offset, alpha_offset) = d2mp_dalpha2[i]; // (alpha, alpha)
            }

            ////////////////////////////////////////////////////////////////////
            // Theta hessian
            ////////////////////////////////////////////////////////////////////
            hessian[theta_offset].template block<3, 3>(0, 0) = presym_block + presym_block.transpose();
            hessian[theta_offset].template block<3, 1>(0, alpha_offset) = d2_theta_dalpha_domega;
            hessian[theta_offset].template block<1, 3>(alpha_offset, 0) = d2_theta_dalpha_domega.transpose();

            hessian[theta_offset](alpha_offset, alpha_offset) = -d1.dot(d2n_dalpha2) - dn_dalpha.dot(d_d1_d_alpha); // (alpha, alpha)

        }
        // Evaluate directional derivative only
        else {
            Vec3 delta_omega;
            Real_ delta_alpha, delta_len;
            unpack_delta_jparams(delta_jparams, len_offset, delta_omega, delta_alpha, delta_len);

            delta_jacobian.setZero();

            const Real_ w_dot_delta_omega       = w.dot(delta_omega);
            const Real_ len_tsX_dot_delta_omega = len * tsX.dot(delta_omega);
            const Real_ psX_dot_delta_omega     = psX.dot(delta_omega);

            ///////////////////////////////////////////////////////////////////
            // e^j hessian
            ///////////////////////////////////////////////////////////////////
            // d^2 e^j / d_omega d_omega
            {
                const Vec3 v = len * tsX;
                if (variation_around_identity) {
                    const Vec3 half_v = 0.5 * v;
                    for (size_t i = 0; i < 3; ++i) {
                        auto dst = delta_jacobian.template block<3, 3>(0, 0);
                        dst = delta_omega * half_v.transpose()
                            - v * delta_omega.transpose();
                        dst.diagonal().array() += 0.5 * len_tsX_dot_delta_omega;
                    }
                }
                else {
                    const Real_ w_dot_v = len * w_dot_tsX;
                    const Mat3 v_cross_term = ropt::cross_product_matrix(tcmsdtc * v);
                    const Vec3 v_cross_w = (-len) * w_cross_tsX;
                    auto dst = delta_jacobian.template block<3, 3>(0, 0);
                    dst = (-w_dot_delta_omega) * v_cross_term +
                          (w_tcm2ptsdtp4 * w_dot_delta_omega) * v.transpose()
                          + ((w_dot_v * eptsmecmftsdtp6 * w + ttcptsm3sdtp5 * v_cross_w - tcmsdtc * v) * w_dot_delta_omega + w_tcm2ptsdtp4 * len_tsX_dot_delta_omega - tcmsdtc * v.cross(delta_omega)) * w.transpose()
                          + delta_omega * (omcdthsq * v + w_dot_v * w_tcm2ptsdtp4).transpose()
                          + (w_dot_v * w_tcm2ptsdtp4 - sinc_th * v - tcmsdtc * v_cross_w) * delta_omega.transpose();
                    dst.diagonal().array() += omcdthsq * len_tsX_dot_delta_omega + w_dot_v * tcm2ptsdtp4 * w_dot_delta_omega;
                }
            }

            delta_jacobian.template block<3, 3>(0, 0) += dtX_domega * delta_len                   // (omega, len)
                                                      +  len * d_t_d_alpha_d_omega * delta_alpha; // (omega, alpha)

            delta_jacobian.template block<3, 1>(0, len_offset) = dtX_domega * delta_omega  // (len, omega)
                                                               + dt_dalpha * delta_alpha;  // (len, alpha)

            delta_jacobian.template block<3, 1>(0, alpha_offset).noalias() += len * (d_t_d_alpha_d_omega * delta_omega) // (alpha, omega)
                                                                           +  len * d2t_dalpha2 * delta_alpha           // (alpha, alpha)
                                                                           +  dt_dalpha * delta_len;                    // (alpha,   len)

            ///////////////////////////////////////////////////////////////////
            // p^j hessian
            ///////////////////////////////////////////////////////////////////
            // d^2 p^j / d_omega d_omega
            {
                const Vec3 v = psX;
                if (variation_around_identity) {
                    const Vec3 half_v = 0.5 * v;
                    for (size_t i = 0; i < 3; ++i) {
                        auto dst = delta_jacobian.template block<3, 3>(midpoint_offset, 0);
                        dst = delta_omega * half_v.transpose()
                            - v * delta_omega.transpose();
                        dst.diagonal().array() += 0.5 * psX_dot_delta_omega;
                    }
                }
                else {
                    const Real_ w_dot_v = w_dot_psX;
                    const Mat3 v_cross_term = ropt::cross_product_matrix(tcmsdtc * v);
                    const Vec3 v_cross_w = - w_cross_psX;
                    auto dst = delta_jacobian.template block<3, 3>(midpoint_offset, 0);
                    dst = (-w_dot_delta_omega) * v_cross_term +
                          (w_tcm2ptsdtp4 * w_dot_delta_omega) * v.transpose()
                          + ((w_dot_v * eptsmecmftsdtp6 * w + ttcptsm3sdtp5 * v_cross_w - tcmsdtc * v) * w_dot_delta_omega + w_tcm2ptsdtp4 * psX_dot_delta_omega - tcmsdtc * v.cross(delta_omega)) * w.transpose()
                          + delta_omega * (omcdthsq * v + w_dot_v * w_tcm2ptsdtp4).transpose()
                          + (w_dot_v * w_tcm2ptsdtp4 - sinc_th * v - tcmsdtc * v_cross_w) * delta_omega.transpose();
                    dst.diagonal().array() += omcdthsq * psX_dot_delta_omega + w_dot_v * tcm2ptsdtp4 * w_dot_delta_omega;
                }
            }

            const Vec3 delta_d_p_d_alpha = d_p_d_alpha_d_omega * delta_omega;

            delta_jacobian.template block<3, 3>(midpoint_offset,          0) += d_p_d_alpha_d_omega * delta_alpha; // (omega, alpha)

            delta_jacobian.template block<3, 1>(midpoint_offset, alpha_offset).noalias() += delta_d_p_d_alpha           // (alpha, omega)
                                                                                         +  d2mp_dalpha2 * delta_alpha; // (alpha, alpha)

            ///////////////////////////////////////////////////////////////////
            // Theta hessian
            ///////////////////////////////////////////////////////////////////

            // std::array<Mat3, 3> d2_n_domega_domega;
            // ropt::hess_rotated_vector(w, ns, d2_n_domega_domega);
            Vec3 d1_dot_delta_n_domega;
            const Real_ ns_dot_delta_omega = ns.dot(delta_omega);
            const Real_ d1_dot_delta_omega = d1.dot(delta_omega);
            {
                if (variation_around_identity) {
                    d1_dot_delta_n_domega = d1 * (0.5 * ns_dot_delta_omega) + (0.5 * d1_dot_delta_omega) * ns;
                }
                else {
                    const Real_ w_dot_d1 = w.dot(d1);
                    const Vec3 tmp2 = tcmsdtc * ns.cross(d1);

                    d1_dot_delta_n_domega  = (w_dot_delta_omega * (w_dot_ns * eptsmecmftsdtp6 * w_dot_d1 - ttcptsm3sdtp5 * w_cross_ns.dot(d1) - tcmsdtc * ns.dot(d1)) + tcm2ptsdtp4 * (ns_dot_delta_omega * w_dot_d1 + d1_dot_delta_omega * w_dot_ns) + tmp2.dot(delta_omega)) * w
                                                + (w_dot_delta_omega * (tcm2ptsdtp4 * w_dot_d1) + omcdthsq * d1_dot_delta_omega) * ns
                                                + d1.dot(d_n_d_omega_w_coeff) * delta_omega
                                                + w_dot_delta_omega * tmp2
                                                + (ns_dot_delta_omega * omcdthsq + w_dot_ns * tcm2ptsdtp4 * w_dot_delta_omega) * d1;
                }
            }
            
            const Vec3 delta_d_theta_d_omega = (presym_block + presym_block.transpose()) * (delta_omega);

            delta_jacobian.template block<1, 3>(theta_offset, 0) = delta_d_theta_d_omega
                                                                 + d2_theta_dalpha_domega * delta_alpha;

            delta_jacobian(theta_offset, alpha_offset) = d2_theta_dalpha_domega.dot(delta_omega) // (alpha, omega)
                                                        + (-d1.dot(d2n_dalpha2) - dn_dalpha.dot(d_d1_d_alpha)) * delta_alpha; // (alpha, alpha)

        }   
    }

    // Fix Eigen alignment issues
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


#endif /* end of include guard: UMBRELLAMESHTERMINALEDGESENSITIVITY_HH */

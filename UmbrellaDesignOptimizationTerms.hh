////////////////////////////////////////////////////////////////////////////////
// UmbrellaDesignOptimizationTerms.hh
////////////////////////////////////////////////////////////////////////////////
#ifndef UMBRELLADESIGNOPTIMIZATIONTERMS_HH
#define UMBRELLADESIGNOPTIMIZATIONTERMS_HH

#include "UmbrellaTargetSurfaceFitter.hh"

template<template<typename> class Object_T>
struct DesignOptimizationObjectTraits;

template<>
struct DesignOptimizationObjectTraits<UmbrellaMesh_T> {
    template<typename T> static size_t     numSimVars      (const UmbrellaMesh_T<T> &obj) { return obj.numDoF(); }
    template<typename T> static size_t     numDesignVars   (const UmbrellaMesh_T<T> &obj) { return obj.numDesignParams(); }
    template<typename T> static size_t     numAugmentedVars(const UmbrellaMesh_T<T> &obj) { return numSimVars(obj) + numDesignVars(obj); }
    template<typename T> static auto       getAugmentedVars(const UmbrellaMesh_T<T> &obj) { return obj.getExtendedDoFsPARL(); }
    template<typename T> static void       setAugmentedVars(UmbrellaMesh_T<T> &obj, const VecX_T<T> &v) { obj.setExtendedDoFsPARL(v); }
    template<typename T> static auto          elasticEnergy(const UmbrellaMesh_T<T> &obj) { return obj.energyElastic(); }
    template<typename T> static auto      gradElasticEnergy(const UmbrellaMesh_T<T> &obj) { return obj.gradientPerArmRestlen(/* updated source */ true, UmbrellaMesh_T<T>::UmbrellaEnergyType::Elastic); }
    template<typename T> static auto applyHessElasticEnergy(const UmbrellaMesh_T<T> &obj, Eigen::Ref<const VecX_T<T>> delta_xp, const HessianComputationMask &mask = HessianComputationMask()) { return obj.applyHessianPerArmRestlen(delta_xp, mask, UmbrellaMesh_T<T>::UmbrellaEnergyType::Elastic); }
};

#include <DesignOptimizationTerms.hh>

template<template<typename> class Object_T>
struct UmbrellaForceObjective : public DesignOptimizationObjectiveTerm<Object_T> {
    using DOT  = DesignOptimizationTerm<Object_T>;
    using DOOT = DesignOptimizationObjectiveTerm<Object_T>;
    using MX3d = Eigen::MatrixX3d;
    using VXd  = Eigen::VectorXd;
    using V3d  = Eigen::Vector3d;

    using JointForces = std::tuple<MX3d, VXd, MX3d>; // full contact force,  separation force and torque acting at each joint

    using ADObject = typename DOT::ADObject;
    using   Object = typename DOT::  Object;

    Real getNormalWeight() const { return m_normalWeight; }
    void setNormalWeight(Real w) {        m_normalWeight = w; this->update(); }

    Real getTangentialWeight() const { return m_tangentialWeight; }
    void setTangentialWeight(Real w) {        m_tangentialWeight = w; this->update(); }
    
    Real getTorqueWeight() const { return m_torqueWeight; }
    void setTorqueWeight(Real w) {        m_torqueWeight = w; this->update(); }

    Real getNormalActivationThreshold() const { return m_epsilon; }
    void setNormalActivationThreshold(Real e) {        m_epsilon = e; this->update(); }

    using DOOT::DOOT;

    template<typename Real_>
    std::tuple<MatX3_T<Real_>, VecX_T<Real_>, MatX3_T<Real_>> umbrellaForces(const Object_T<Real_> &obj) const {
        const size_t nu = obj.UmbrellaMesh_T<Real_>::numUmbrellas();
        std::tuple<MatX3_T<Real_>, VecX_T<Real_>, MatX3_T<Real_>> result;
        auto &forces           = std::get<0>(result);
        auto &separationForces = std::get<1>(result);
        auto &torques          = std::get<2>(result);

        forces          .resize(nu * 2, 3);
        separationForces.resize(nu * 2);
        torques         .resize(nu * 2, 3);

        VecX_T<Real_> rf = obj.rivetForces();
        for (size_t ui = 0; ui < nu; ++ui) {
            std::array<size_t, 2> uji{{ obj.UmbrellaMesh_T<Real_>::getUmbrellaCenterJi(ui, 0), obj.UmbrellaMesh_T<Real_>::getUmbrellaCenterJi(ui, 1)}};
            for (size_t i = 0; i < 2; ++ i) {
                const auto &j = obj.joint(uji[i]);
                Vec3_T<Real_> f = rf.template segment<3>(obj.dofOffsetForJoint(uji[i]));
                forces.row(ui * 2 + i) = f;
                separationForces[ui * 2 + i] = j.normal().dot(f);
                torques.row(ui * 2 + i) = rf.template segment<3>(obj.dofOffsetForJoint(uji[i]) + 3);
            }
        }
        return result;
    }

    const JointForces &umbrellaForces() const {
        if (!m_cachedForces) m_cachedForces = std::make_unique<JointForces>(umbrellaForces(m_obj));
        return *m_cachedForces;
    }

protected:
    using DOT::m_obj;
    mutable std::unique_ptr<JointForces> m_cachedForces;

    Real m_normalWeight = 1.0, m_tangentialWeight = 0.0, m_torqueWeight = 0.0,
         m_epsilon = 0.0; // activation threshold above which the separation force is penalized; make this negative to ensure removal of separation forces.


    // (Optional) update cached quantities shared between energy/gradient calculations.
    virtual void m_update() override {
        m_cachedForces.reset();
        m_cachedForces = std::make_unique<JointForces>(umbrellaForces());
    }

    std::tuple<VXd, VXd, VXd> getJointWeights() const {
        std::tuple<VXd, VXd, VXd> result;
        const size_t nu = m_obj.UmbrellaMesh_T<Real>::numUmbrellas();
        std::get<0>(result).setConstant(nu * 2,     m_normalWeight);
        std::get<1>(result).setConstant(nu * 2, m_tangentialWeight);
        std::get<2>(result).setConstant(nu * 2,     m_torqueWeight);
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Implementation of DesignOptimizationTerm interface
    ////////////////////////////////////////////////////////////////////////////
    virtual Real m_value() const override {
        MX3d forces;
        VXd separationForces;
        MX3d torques;
        std::tie(forces, separationForces, torques) = umbrellaForces();

        VXd normalWeightsSqrt, tangentialWeightsSqrt, torqueWeightsSqrt;
        std::tie(normalWeightsSqrt, tangentialWeightsSqrt, torqueWeightsSqrt) = getJointWeights();

        normalWeightsSqrt = normalWeightsSqrt.cwiseSqrt();
        tangentialWeightsSqrt = tangentialWeightsSqrt.cwiseSqrt();
        torqueWeightsSqrt = torqueWeightsSqrt.cwiseSqrt();

        return 0.5 * (     normalWeightsSqrt.array() * (separationForces.array() - m_epsilon).cwiseMax(0.0)).matrix().squaredNorm()
            +  0.5 * ((tangentialWeightsSqrt.asDiagonal() * forces).squaredNorm() - (tangentialWeightsSqrt.asDiagonal() * separationForces).squaredNorm())
            +  0.5 * (     torqueWeightsSqrt.asDiagonal() * torques).squaredNorm();
    }

    template<typename Real_>
    VecX_T<Real_> m_grad_impl(const Object_T<Real_> &obj) const {
        const size_t nu = obj.UmbrellaMesh_T<Real_>::numUmbrellas();

        HessianComputationMask mask;
        mask.skipBRods = true;
        mask.designParameter_in = false;

        MatX3_T<Real_> forces;
        VecX_T<Real_> separationForces;
        MatX3_T<Real_> torques;
        std::tie(forces, separationForces, torques) = umbrellaForces(obj);

        VXd normalWeights, tangentialWeights, torqueWeights;
        std::tie(normalWeights, tangentialWeights, torqueWeights) = getJointWeights();

        // dJ/df df/dxp term
        // (Variation due to derivative of joint forces, holding normal fixed)
        VecX_T<Real_> dJ_df;
        dJ_df.setZero(this->numVars());
        for (size_t ui = 0; ui < nu; ++ui) {
            std::array<size_t, 2> uji{{ obj.UmbrellaMesh_T<Real_>::getUmbrellaCenterJi(ui, 0), obj.UmbrellaMesh_T<Real_>::getUmbrellaCenterJi(ui, 1)}};
            for (size_t i = 0; i < 2; ++ i) {
                const auto &j = obj.joint(uji[i]);
                dJ_df.template segment<3>(obj.dofOffsetForJoint(uji[i])) =
                        normalWeights[ui * 2 + i] * std::max<Real_>(separationForces[ui * 2 + i] - m_epsilon, 0.0) * j.normal() +
                    tangentialWeights[ui * 2 + i] * (forces.row(ui * 2 + i).transpose() - separationForces[ui * 2 + i] * j.normal());
                dJ_df.template segment<3>(obj.dofOffsetForJoint(uji[i])+3) = 
                        torqueWeights[ui * 2 + i] * torques.row(ui * 2 + i).transpose();
            }
        }
        VecX_T<Real_> result = -obj.UmbrellaMesh_T<Real_>::applyHessianPerArmRestlen(dJ_df, mask, UmbrellaMesh_T<Real_>::UmbrellaEnergyType::Elastic); // rivet forces are -dE/dx; explicitly use the elastic energy hessian only.

        // dJ/dn term
        // (Derivative with respect to normal, holding joint forces fixed)
        for (size_t ui = 0; ui < nu; ++ui) {
            std::array<size_t, 2> uji{{ obj.UmbrellaMesh_T<Real_>::getUmbrellaCenterJi(ui, 0), obj.UmbrellaMesh_T<Real_>::getUmbrellaCenterJi(ui, 1)}};
            for (size_t i = 0; i < 2; ++ i) {
                const auto &j = obj.joint(uji[i]);
                result.template segment<3>(obj.dofOffsetForJoint(uji[i]) + 3) +=
                    (     normalWeights[ui * 2 + i] * std::max<Real_>(separationForces[ui * 2 + i] - m_epsilon, 0.0) * forces.row(ui * 2 + i)
                    - tangentialWeights[ui * 2 + i] *                 separationForces[ui * 2 + i]                   * forces.row(ui * 2 + i)) *
                        rotation_optimization<Real_>::grad_rotated_vector(j.omega(), j.source_normal());
            }
        }

        return result;
    }
    virtual VXd m_grad() const override { return m_grad_impl(m_obj); }

    virtual VXd m_delta_grad(Eigen::Ref<const VXd> /* delta_xp */, const ADObject &autodiffObject) const override {
        return extractDirectionalDerivative(m_grad_impl(autodiffObject));
    }
};

// Wrapper for the deployment force objective.
template<template<typename> class Object_T>
struct DeploymentForceDOOT : public DesignOptimizationObjectiveTerm<Object_T> {
    using DOT      = DesignOptimizationTerm<Object_T>;
    using DOOT     = DesignOptimizationObjectiveTerm<Object_T>;
    using Object   = typename DOT::Object;
    using ADObject = typename DOT::ADObject;
    using VXd      = Eigen::VectorXd;
    using Vec3     = Vec3_T<Real>;

    DeploymentForceDOOT(const Object &obj)
        : DOOT(obj) { }

    using DOOT::weight;

    Real getActivationThreshold() const { return m_epsilon; }
    void setActivationThreshold(Real e) {        m_epsilon = e; this->update(); }

protected:
    using DOT::m_obj;
    Real m_epsilon = 0.0; // activation threshold above which the force is penalized; make this negative to ensure removal of tensile forces.

    ////////////////////////////////////////////////////////////////////////////
    // Implementation of DesignOptimizationTerm interface
    ////////////////////////////////////////////////////////////////////////////

    virtual Real m_value() const override {
        return 0.5 * ((m_obj.getUmbrellaHeights().array() - m_epsilon).cwiseMax(0.0)).matrix().squaredNorm();
    }

    virtual VXd m_grad() const override {
        VXd result = VXd::Zero(this->numVars());
        VXd umbrellaHeights = m_obj.getUmbrellaHeights();
        for (size_t ui = 0; ui < m_obj.numUmbrellas(); ++ui) {
            size_t    top_ji = m_obj.getUmbrellaCenterJi(ui, 0);
            size_t bottom_ji = m_obj.getUmbrellaCenterJi(ui, 1);
            Real coeff = umbrellaHeights[ui] - m_epsilon;
            if (coeff < 0) continue;
            Vec3 tangent = (m_obj.joint(top_ji).pos() - m_obj.joint(bottom_ji).pos()).normalized();
            result.template segment<3>(m_obj.dofOffsetForJoint(top_ji)) += coeff * tangent;
            result.template segment<3>(m_obj.dofOffsetForJoint(bottom_ji)) += - coeff * tangent;
        }
        return result;
    }

    virtual VXd m_delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &/* ado */) const override {
        VXd result = VXd::Zero(delta_xp.size());
        using M3d = Mat3_T<Real>;
        VXd umbrellaHeights = m_obj.getUmbrellaHeights();
        for (size_t ui = 0; ui < m_obj.numUmbrellas(); ++ ui) {
            size_t    top_ji     = m_obj.getUmbrellaCenterJi(ui, 0);
            size_t bottom_ji     = m_obj.getUmbrellaCenterJi(ui, 1);
            size_t    top_ji_dof = m_obj.dofOffsetForJoint(   top_ji);
            size_t bottom_ji_dof = m_obj.dofOffsetForJoint(bottom_ji);
            Real height = umbrellaHeights[ui];
            if (height == 0) std::cout<<"Distance Deployment Hessian term encountered zero height umbrella at joint "<<ui<<"!"<<std::endl;
            // If the height is below the activation threshold, then the term is zero.
            if (height < m_epsilon) continue;
            
            Real coeff_one = 1.0 - m_epsilon / height;
            Real coeff_two = 1.0 - coeff_one;
            Vec3 tangent = (m_obj.joint(top_ji).pos() - m_obj.joint(bottom_ji).pos()) / height;
            M3d hessianBlock;
            hessianBlock = coeff_one * M3d::Identity() + (coeff_two * tangent) * tangent.transpose();
            
            result.template segment<3>(top_ji_dof) += hessianBlock * delta_xp.template segment<3>(top_ji_dof) - hessianBlock * delta_xp.template segment<3>(bottom_ji_dof);
            result.template segment<3>(bottom_ji_dof) += hessianBlock * delta_xp.template segment<3>(bottom_ji_dof) - hessianBlock * delta_xp.template segment<3>(top_ji_dof);
        }
        return result;
    }
};

#endif /* end of include guard: UMBRELLADESIGNOPTIMIZATIONTERMS_HH */

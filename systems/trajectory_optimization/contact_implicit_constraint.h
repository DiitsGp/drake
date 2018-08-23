#pragma once

#include <memory>
#include <vector>

#include "drake/math/autodiff.h"
#include "drake/multibody/kinematics_cache_helper.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/decision_variable.h"

namespace drake {
namespace systems {
namespace trajectory_optimization {

class ContactImplicitConstraint
	: public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactImplicitConstraint)

  ContactImplicitConstraint(
  		const RigidBodyTree<double>& tree,
      const RigidBodyTree<double>& empty_tree,
  		std::shared_ptr<plants::KinematicsCacheWithVHelper<AutoDiffXd>>
  			kinematics_cache_with_v_helper, int num_lambda, double elasticity, double tol);

  ~ContactImplicitConstraint() override {}

  template <typename DerivedQL, typename DerivedV,
            typename DerivedLambda>
  typename std::enable_if<is_eigen_vector_of<DerivedQL, drake::symbolic::Variable>::value &&
                             is_eigen_vector_of<DerivedV, drake::symbolic::Variable>::value&&
                              is_eigen_vector_of<DerivedLambda, drake::symbolic::Variable>::value,
                          Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, 1>>::type
  CompositeEvalInput(const Eigen::MatrixBase<DerivedQL>& q,
                     const Eigen::MatrixBase<DerivedV>& v,
                     const Eigen::MatrixBase<DerivedLambda>& lambda) const {
    DRAKE_ASSERT(q.rows() == num_positions_);
    DRAKE_ASSERT(v.rows() == num_velocities_);
    DRAKE_ASSERT(lambda.rows() == num_lambda_);
    Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, 1> x(num_vars(), 1);
    x << q, v, lambda;
    //x << q, v;
    return x;
  }

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd& y) const override;
 private:
  const RigidBodyTree<double>* tree_;
  const RigidBodyTree<double>* empty_tree_;
  const int num_positions_;
  const int num_velocities_;
  const int num_lambda_;
  const int num_contacts_;
  const double elasticity_;
  const double tol_;

  std::shared_ptr<plants::KinematicsCacheWithVHelper<AutoDiffXd>>
  	  kinematics_cache_with_v_helper_;
};


// Collision and stuf?
class TimestepIntegrationConstraint
  : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TimestepIntegrationConstraint)

  TimestepIntegrationConstraint(
      const RigidBodyTree<double>& tree,
      const RigidBodyTree<double>& empty_tree,
      std::shared_ptr<plants::KinematicsCacheWithVHelper<AutoDiffXd>>
        kinematics_cache_with_v_helper, int num_lambda, double elasticity);
  ~TimestepIntegrationConstraint() {}

  template <typename DerivedQL, typename DerivedV,
            typename DerivedLambda>
  typename std::enable_if<is_eigen_vector_of<DerivedQL, drake::symbolic::Variable>::value &&
                             is_eigen_vector_of<DerivedV, drake::symbolic::Variable>::value&&
                              is_eigen_vector_of<DerivedLambda, drake::symbolic::Variable>::value,
                          Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, 1>>::type
  CompositeEvalInput(const Eigen::MatrixBase<DerivedQL>& q,
                     const Eigen::MatrixBase<DerivedV>& v,
                     const Eigen::MatrixBase<DerivedLambda>& lambda) const {
    DRAKE_ASSERT(q.rows() == num_positions_);
    DRAKE_ASSERT(v.rows() == num_velocities_);
    //DRAKE_ASSERT(lambda.rows() == num_lambda_);
    Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, 1> x(num_vars(), 1);
    x << q, v, lambda;
    //x << q, v;
    return x;
  }

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd& y) const;
 private:
  const RigidBodyTree<double>* tree_;
  const RigidBodyTree<double>* empty_tree_;
  const int num_positions_;
  const int num_velocities_;
  const int num_lambda_;
  const int num_contacts_;
  const double elasticity_;

  std::shared_ptr<plants::KinematicsCacheWithVHelper<AutoDiffXd>>
      kinematics_cache_with_v_helper_;
};


} // namespace trajectory_optimization
} // namespace systems
} // namespace drake
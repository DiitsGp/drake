#include "drake/systems/trajectory_optimization/elastic_contact_implicit_direct_transcription.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "drake/common/eigen_types.h"
#include "drake/common/unused.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/systems/trajectory_optimization/rigid_body_tree_multiple_shooting_internal.h"
#include "drake/systems/trajectory_optimization/position_constraint_force_evaluator.h"

namespace drake {
namespace systems {
namespace trajectory_optimization {
using plants::KinematicsCacheWithVHelper;
using plants::KinematicsCacheHelper;
using Eigen::Dynamic;
/**
 * Implements the constraint for the backward Euler integration
 * <pre>
 * qᵣ - qₗ = q̇ᵣ*h
 * Mᵣ(vᵣ - vₗ) = (B*uᵣ + Jᵣᵀ*λᵣ -c(qᵣ, vᵣ))h
 * </pre>
 * where
 * qᵣ: The generalized position on the right knot.
 * qₗ: The generalized position on the left knot.
 * vᵣ: The generalized velocity on the right knot.
 * vₗ: The generalized velocity on the left knot.
 * uᵣ: The actuator input on the right knot.
 * Mᵣ: The inertia matrix computed from qᵣ.
 * λᵣ: The constraint force (e.g., contact force, joint limit force, etc) on the
 * right knot.
 * c(qᵣ, vᵣ): The Coriolis, gravity and centripedal force on the right knot.
 * h: The duration between the left and right knot.
 */
CustomDirectTranscriptionConstraint::CustomDirectTranscriptionConstraint(
    const RigidBodyTree<double>& tree,
    std::shared_ptr<KinematicsCacheWithVHelper<AutoDiffXd>> kinematics_helper)
    : Constraint(tree.get_num_positions() + tree.get_num_velocities(),
                 1 + 2 * tree.get_num_positions() +
                     4 * tree.get_num_velocities() + tree.get_num_actuators(),
                 Eigen::VectorXd::Zero(tree.get_num_positions() +
                                       tree.get_num_velocities()),
                 Eigen::VectorXd::Zero(tree.get_num_positions() +
                                       tree.get_num_velocities())),
      tree_(&tree),
      num_positions_{tree.get_num_positions()},
      num_velocities_{2*tree.get_num_velocities()},
      num_actuators_{tree.get_num_actuators()},
      num_lambda_{0},
      kinematics_helper1_{kinematics_helper} {}

void CustomDirectTranscriptionConstraint::AddGeneralizedConstraintForceEvaluator(
    std::unique_ptr<GeneralizedConstraintForceEvaluator> evaluator) {
  set_num_vars(num_vars() + evaluator->lambda_size());
  num_lambda_ += evaluator->lambda_size();
  generalized_constraint_force_evaluators_.push_back(std::move(evaluator));
}

void CustomDirectTranscriptionConstraint::AddGeneralizedConstraintForceEvaluator(
    std::unique_ptr<GeneralizedConstraintForceEvaluator> evaluator,
    const Eigen::Ref<const solvers::VectorXDecisionVariable>& new_lambda_vars,
    solvers::VectorXDecisionVariable* x) {
  if (x->rows() != num_vars()) {
    throw std::runtime_error("x doesn't have the right size.");
  }
  if (new_lambda_vars.rows() != evaluator->lambda_size()) {
    throw std::runtime_error("new_lambda_vars doesn't have the right size.");
  }
  AddGeneralizedConstraintForceEvaluator(std::move(evaluator));
  // Append new_lambda_vars to the end of x, since lambda_r shows up in the end
  // of x, as defined in CompositeEvalInput(...)
  x->conservativeResize(num_vars());
  x->tail(new_lambda_vars.rows()) = new_lambda_vars;
}

void CustomDirectTranscriptionConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd& y) const {
  DRAKE_ASSERT(x.size() == num_vars());

  AutoDiffVecXd y_t;
  Eval(math::initializeAutoDiff(x), y_t);
  y = math::autoDiffToValueMatrix(y_t);
}

void CustomDirectTranscriptionConstraint::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const {
  DRAKE_ASSERT(x.size() == num_vars());

  int x_count = 0;
  // A lambda expression to take num_element entreis from x, in a certain
  // order.
  auto x_segment = [x, &x_count](int num_element) {
    x_count += num_element;
    return x.segment(x_count - num_element, num_element);
  };

  const AutoDiffXd h = x(0); // IMPORTANT this is $\Delta(h)$
  x_count++;
  const AutoDiffVecXd q_l = x_segment(num_positions_);
  const AutoDiffVecXd v_plus_l = x_segment(num_velocities_/2);
  const AutoDiffVecXd v_minus_l = x_segment(num_velocities_/2);
  const AutoDiffVecXd q_r = x_segment(num_positions_);
  const AutoDiffVecXd v_plus_r = x_segment(num_velocities_/2);  
  const AutoDiffVecXd v_minus_r = x_segment(num_velocities_/2);
  const AutoDiffVecXd u_r = x_segment(num_actuators_);
  //const AutoDiffVecXd lambda_r = x_segment(num_lambda_);

  auto kinsol = kinematics_helper1_->UpdateKinematics(q_r, v_minus_r);
  y.resize(num_constraints());
  AutoDiffVecXd y_pos, y_dyn;
  // y = [q; vminus; vplus]

  // By using backward Euler integration, the constraint is
  // qᵣ - qₗ = q̇ᵣ*h
  // Mᵣ(vᵣ - vₗ) = (B*uᵣ + Jᵣᵀ*λᵣ -c(qᵣ, vᵣ))h
  const MatrixX<AutoDiffXd> map_v_to_qdot =
      RigidBodyTree<double>::GetVelocityToQDotMapping(kinsol);

  const AutoDiffVecXd qdot_minus_r = map_v_to_qdot * v_minus_r;
  const AutoDiffVecXd qdot_plus_l = map_v_to_qdot * v_plus_l;
  // TODO(hongkai.dai): Project qdot_r to the constraint manifold (for example,
  // if q contains unit quaternion, and we need to project this backward Euler
  // integration on the unit quaternion manifold.)
  y_pos = q_r - q_l - qdot_minus_r* (h);

  const auto M = tree_->massMatrix(kinsol);

  // Compute the Coriolis force, centripedal force, etc.
  const typename RigidBodyTree<AutoDiffXd>::BodyToWrenchMap
      no_external_wrenches;
  const auto c = tree_->dynamicsBiasTerm(kinsol, no_external_wrenches);

  // int lambda_count = 0;
  // for (const auto& evaluator : generalized_constraint_force_evaluators_) {
  //   AutoDiffVecXd q_v_lambda(num_positions_ + num_velocities_/2 +
  //                            evaluator->lambda_size());
  //   q_v_lambda << q_r, v_minus_r,
  //       lambda_r.segment(lambda_count, evaluator->lambda_size());
  //   AutoDiffVecXd generalized_constraint_force(num_velocities_/2);
  //   std::cerr<<q_v_lambda.rows()<<std::endl;
  //   evaluator->Eval(q_v_lambda, generalized_constraint_force);
  //   total_generalized_constraint_force += generalized_constraint_force;
  //   lambda_count += evaluator->lambda_size();
  // }

  // M*v_minus_r = M*v_plus_l + Bu-c + J^T*lambda
  y_dyn = M * (v_minus_r - v_plus_l) +
      (c - (tree_->B * u_r)) * h;

  y << y_pos, y_dyn;
}


ElasticContactImplicitDirectTranscription::ElasticContactImplicitDirectTranscription(
    const RigidBodyTree<double>& tree,
    const RigidBodyTree<double>& empty_tree,
    int num_time_samples,
    double minimum_timestep, double maximum_timestep, int num_contact_lambda,
    double compl_tol, double elasticity)
    : MultipleShooting(tree.get_num_actuators(),
                       tree.get_num_positions() + 2*tree.get_num_velocities() + num_contact_lambda,
                       num_time_samples, minimum_timestep, maximum_timestep),
      tree_{&tree},
      empty_tree_{&empty_tree},
      num_positions_{tree.get_num_positions()},
      num_velocities_{2*tree.get_num_velocities()},
      num_lambda_{num_contact_lambda},
      compl_tol_{compl_tol},
      elasticity_{elasticity},
      lambda_vars_(NewContinuousVariables(
          num_contact_lambda, N(), "contact_lambda")) {
  // For each knot, we will need to impose a transcription/collocation
  // constraint. Each of these constraints require us caching some
  // kinematics info.

  dtc_kinematics_cache_with_v_helpers_.resize(num_time_samples);
  dtc_kinematics_cache_helpers_.resize(num_time_samples);
  tic_kinematics_cache_with_v_helpers_.resize(num_time_samples);
  cic_kinematics_cache_with_v_helpers_.resize(num_time_samples);

  for (int i = 0; i < num_time_samples; ++i) {
    dtc_kinematics_cache_with_v_helpers_[i] =
        std::make_shared<KinematicsCacheWithVHelper<AutoDiffXd>>(*tree_);
    dtc_kinematics_cache_helpers_[i] =
        std::make_shared<KinematicsCacheHelper<AutoDiffXd>>(*tree_);
    tic_kinematics_cache_with_v_helpers_[i] = 
        std::make_shared<KinematicsCacheWithVHelper<AutoDiffXd>>(*tree_);
    cic_kinematics_cache_with_v_helpers_[i] = 
        std::make_shared<KinematicsCacheWithVHelper<AutoDiffXd>>(*tree_);
  }

  q_vars_.resize(num_positions_, N());
  v_vars_.resize(num_velocities_, N());
  for (int i = 0; i < N(); ++i) {
    q_vars_.col(i) = x_vars().segment(num_states() * i, num_positions_);
    v_vars_.col(i) =
        x_vars().segment(num_states() * i + num_positions_, num_velocities_);
  }

  direct_transcription_constraints_.reserve(N() - 1);
  for (int i = 0; i < N() - 1; ++i) {
    auto transcription_cnstr = std::make_shared<CustomDirectTranscriptionConstraint>(
        *tree_, dtc_kinematics_cache_with_v_helpers_[i + 1]);
    // Add RigidBodyConstraint::PositionConstraint to the constraint force Jᵀλ
    // used in the dynamics for direct transcription.
    auto position_constraint_force_evaluator =
        std::make_unique<PositionConstraintForceEvaluator>(
            *tree_, dtc_kinematics_cache_helpers_[i + 1]);
    transcription_cnstr->AddGeneralizedConstraintForceEvaluator(
        std::move(position_constraint_force_evaluator));

    const solvers::VectorXDecisionVariable transcription_vars =
        transcription_cnstr->CompositeEvalInput(
            h_vars()(i), q_vars_.col(i), v_vars_.col(i), q_vars_.col(i + 1),
            v_vars_.col(i + 1),
            u_vars().segment((i + 1) * num_inputs(), num_inputs()));
    direct_transcription_constraints_.emplace_back(transcription_cnstr,
                                                   transcription_vars);
  }

  timestep_integration_constraints_.reserve(N());
  for (int i = 0; i < N(); i++) {
    auto timestep_cnstr = std::make_shared<TimestepIntegrationConstraint>(
        *tree_, *empty_tree_,
        tic_kinematics_cache_with_v_helpers_[i], num_lambda_, elasticity_);
    const solvers::VectorXDecisionVariable timestep_cnstr_vars =
        timestep_cnstr->CompositeEvalInput(
          q_vars_.col(i), v_vars_.col(i), lambda_vars_.col(i));
    timestep_integration_constraints_.emplace_back(timestep_cnstr, timestep_cnstr_vars);
  }

  contact_implicit_constraints_.reserve(N());
  for (int i = 0; i < N(); i++) {
    auto contact_implicit_cnstr = std::make_shared<ContactImplicitConstraint>(
      *tree_, *empty_tree_,
      cic_kinematics_cache_with_v_helpers_[i], num_lambda_, elasticity_, compl_tol_);
    const solvers::VectorXDecisionVariable contact_implicit_cnstr_vars = 
        contact_implicit_cnstr->CompositeEvalInput(
          q_vars_.col(i), v_vars_.col(i), lambda_vars_.col(i));
    contact_implicit_constraints_.emplace_back(contact_implicit_cnstr, contact_implicit_cnstr_vars);
  }
}

void ElasticContactImplicitDirectTranscription::Compile() {
  for (int i = 0; i < N() - 1; ++i) {
    AddConstraint(direct_transcription_constraints_[i].constraint(),
                  direct_transcription_constraints_[i].variables());
  }

  for (int i = 0; i < N(); i++) {
    AddConstraint(timestep_integration_constraints_[i].constraint(),
                  timestep_integration_constraints_[i].variables());
  }

  for (int i = 0; i < N(); i++) {
    AddConstraint(contact_implicit_constraints_[i].constraint(),
                  contact_implicit_constraints_[i].variables());
  }
  
}

void ElasticContactImplicitDirectTranscription::
    AddGeneralizedConstraintForceEvaluatorToTranscription(
        int interval_index,
        std::unique_ptr<GeneralizedConstraintForceEvaluator> evaluator,
        const Eigen::Ref<const solvers::VectorXDecisionVariable>&
            evaluator_lambda) {
  DRAKE_ASSERT(evaluator->lambda_size() == evaluator_lambda.rows());
  solvers::VectorXDecisionVariable vars =
      direct_transcription_constraints_[interval_index].variables();
  auto direct_transcription_constraint =
      direct_transcription_constraints_[interval_index].constraint();
  direct_transcription_constraint->AddGeneralizedConstraintForceEvaluator(
      std::move(evaluator), evaluator_lambda, &vars);
  // Now update the Binding in direct_transcription_constraints_
  direct_transcription_constraints_[interval_index] =
      solvers::Binding<CustomDirectTranscriptionConstraint>(
          direct_transcription_constraint, vars);
}





void ElasticContactImplicitDirectTranscription::DoAddRunningCost(
    const symbolic::Expression& g) {
  // Add the running cost ∫ g(t, x, u)
  // We discretize this continuous integration as
  // sum_{i = 0, ..., N - 2} h_i * g_{i+1}
  for (int i = 0; i < N() - 2; ++i) {
    AddCost(SubstitutePlaceholderVariables(g * h_vars()(i), i + 1));
  }
}

PiecewisePolynomialTrajectory
ElasticContactImplicitDirectTranscription::ReconstructStateTrajectory() const {
  Eigen::VectorXd times = GetSampleTimes();
  std::vector<double> times_vec(N());
  std::vector<Eigen::MatrixXd> states(N());

  for (int i = 0; i < N(); ++i) {
    times_vec[i] = times(i);
    states[i] = GetSolution(state(i));
  }
  return PiecewisePolynomialTrajectory(
      PiecewisePolynomial<double>::FirstOrderHold(times_vec, states));
}

PiecewisePolynomialTrajectory
ElasticContactImplicitDirectTranscription::ReconstructInputTrajectory() const {
  Eigen::VectorXd times = GetSampleTimes();
  std::vector<double> times_vec(N());
  std::vector<Eigen::MatrixXd> inputs(N());

  for (int i = 0; i < N(); ++i) {
    times_vec[i] = times(i);
    inputs[i] = GetSolution(input(i));
  }
  return PiecewisePolynomialTrajectory(
      PiecewisePolynomial<double>::ZeroOrderHold(times_vec, inputs));
}

}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake

#include "drake/systems/trajectory_optimization/contact_implicit_constraint.h"

#include "drake/multibody/kinematics_cache.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace systems {
namespace trajectory_optimization {

const bool debug = false;

void CollisionStuff(
    const RigidBodyTree<double>& tree,
    const RigidBodyTree<double>& empty_tree,
    const VectorX<AutoDiffXd>& q,
    const VectorX<AutoDiffXd>& v_minus,
    const MatrixX<AutoDiffXd>& lambda,
    int nx,
    double elasticity,
    VectorX<AutoDiffXd>* phi_out,
    MatrixX<AutoDiffXd>* Jphi_out,
    VectorX<AutoDiffXd>* lambda_out,
    MatrixX<double>* tensornormal_out) {

  auto kinsol = tree.doKinematics(q);

  RigidBody<double>* brick = tree.FindBody("contact_implicit_brick");
  const auto& contacts_B = brick->get_contact_points();
  const KinematicsCache<double> cache = tree.doKinematics(
        math::autoDiffToValueMatrix(q));
  const int brick_index = brick->get_body_index();

  const int num_lambda_ = lambda.rows();
  const int num_contacts_ = num_lambda_ / 3;

  // `B` is body frame. `W` is world frame.
  const auto contacts_W = tree.transformPoints(
      cache, contacts_B, brick_index, 0 /* world */);

  Eigen::VectorXd phi;
  // `L` is the local frame of whatever bodies in the world.
  Eigen::Matrix3Xd normal, world_contacts_W, world_contacts_L;
  std::vector<int> body_idx;

  // Do collision without body in the way, so that points are not attracted to
  // it.
  // TODO: Just project other stuff away?
  const KinematicsCache<double> scene_cache = empty_tree.doKinematics(
      empty_tree.getZeroConfiguration());
  const_cast<RigidBodyTree<double>&>(empty_tree).collisionDetectFromPoints(
      scene_cache, contacts_W,
      phi, normal, world_contacts_W, world_contacts_L, body_idx, false);

  // TODO: Need to remap body indices.

  Eigen::VectorXi idx_a(num_contacts_), idx_b(num_contacts_);
  for (size_t i = 0; i < body_idx.size(); i++) {
    idx_a[i] = brick_index;
    idx_b[i] = body_idx[i];
  }

  // `J` is 3*nc x nq. Projecting along normals yields Jphi.
  MatrixX<AutoDiffXd> J;
  tree.computeContactJacobians(
      kinsol, idx_a, idx_b, contacts_B, world_contacts_W, J);

  // Project along normals.
  MatrixX<double>& tensornormal = *tensornormal_out;
  tensornormal = Eigen::MatrixXd::Zero(num_lambda_, num_contacts_);
  for (int i = 0; i < num_contacts_; i++) {
    for (int j = 0; j < 3; j++) {
      tensornormal(3*i + j, i) = normal(j, i);
    }
  }

  *Jphi_out = tensornormal.cast<AutoDiffXd>().transpose() * J;

  const int nq = q.rows();
  MatrixX<double> Jphi_x(num_contacts_, nx);
  const int offset_q_in_x = 0;
  Jphi_x.setZero();
  Jphi_x.middleCols(offset_q_in_x, nq) = math::autoDiffToValueMatrix(*Jphi_out);

  // Ensure the derivative of `phi` is w.r.t. `x`, not `q`.
  *phi_out = math::initializeAutoDiffGivenGradientMatrix(
      phi, Jphi_x);

  const MatrixX<AutoDiffXd> map_v_to_qdot =
      RigidBodyTree<double>::GetVelocityToQDotMapping(kinsol);
  const AutoDiffVecXd qd_minus = map_v_to_qdot * v_minus;
  const auto M = tree.massMatrix(kinsol);
  // this line is broken
  const auto projected_mass = ((*Jphi_out) * M.inverse() * (*Jphi_out).transpose()).inverse();
  *lambda_out = -(1+elasticity) * 
                projected_mass *
                (*Jphi_out) * qd_minus;

  if (debug) {
    using namespace std;
    cout << "projected_mass" << M << endl;
    cout << "lambda: " << lambda.transpose() << endl;
    cout << "lambda_calc: " << (*lambda_out).transpose() << endl;
    //std::cerr << "contacts_B:\n" << contacts_B << std::endl;;
    //cout << "contacts_W:\n" << contacts_W << "\n---\n";
    //cout << "world_contacts_W:\n" << world_contacts_W << "\n---\n";
  }
}

ContactImplicitConstraint::ContactImplicitConstraint(
    const RigidBodyTree<double>& tree,
    const RigidBodyTree<double>& empty_tree,
    std::shared_ptr<plants::KinematicsCacheWithVHelper<AutoDiffXd>>
        kinematics_cache_with_v_helper, int num_lambda, double elasticity, double tol)
/*
 * Original:
 * 2*num_lambda/3 + 1,
 * tree.get_num_positions() + 2*tree.get_num_velocities() + num_lambda
 */
    : Constraint(2*num_lambda/3 + 1, tree.get_num_positions() +
                  2*tree.get_num_velocities() + num_lambda,
        Eigen::MatrixXd::Zero(2*num_lambda/3 + 1, 1),
        Eigen::MatrixXd::Constant(2*num_lambda/3 + 1, 1, tol)), 
      tree_(&tree),
      empty_tree_(&empty_tree),
      num_positions_{tree.get_num_positions()},
      num_velocities_{2*tree.get_num_velocities()},
      num_lambda_{num_lambda},
      num_contacts_{num_lambda/3},
      elasticity_{elasticity},
      tol_{tol},
      kinematics_cache_with_v_helper_{kinematics_cache_with_v_helper} {
    unused(num_contacts_);
    Eigen::VectorXd UB(2*num_contacts_ + 1, 1);
    UB.head(2*num_contacts_) = Eigen::MatrixXd::Constant(
                2*num_contacts_, 1, std::numeric_limits<double>::infinity());
    UB.tail(1) = Eigen::MatrixXd::Constant(1, 1, tol_);
    UpdateUpperBound(UB);
}

void ContactImplicitConstraint::DoEval(
  const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd& y) const {
  AutoDiffVecXd y_t;
  Eval(math::initializeAutoDiff(x), y_t);
  y = math::autoDiffToValueMatrix(y_t);
}

void ContactImplicitConstraint::DoEval(
  const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const {
  y.resize(num_constraints());

  int x_count = 0;

  auto x_segment = [x, &x_count](int num_element) {
    x_count += num_element;
    return x.segment(x_count - num_element, num_element);
  };

  const AutoDiffVecXd q = x_segment(num_positions_);
  const AutoDiffVecXd v_plus = x_segment(num_velocities_/2);
  const AutoDiffVecXd v_minus = x_segment(num_velocities_/2);
  const AutoDiffVecXd lambda = x_segment(num_lambda_);

  VectorX<AutoDiffXd> phi;
  MatrixX<AutoDiffXd> Jphi;
  VectorX<AutoDiffXd> lambda_calc;
  MatrixX<double> tensornormal;
  CollisionStuff(
      *tree_, *empty_tree_, q, v_minus,lambda,
      x.size(), elasticity_,
      &phi, &Jphi, &lambda_calc, &tensornormal);

  // Ensure that 
  // MatrixX<AutoDiffXd> autotensornormal(
  //     tensornormal.rows(), tensornormal.cols());
  // VectorX<double> dx0 = Eigen::VectorXd::Zero(x.rows());
  // for (int r = 0; r < autotensornormal.rows(); ++r) {
  //   for (int c = 0; c < autotensornormal.cols(); ++c) {
  //     autotensornormal(r, c) = tensornormal(r, c);
  //     autotensornormal(r, c).derivatives() = dx0;
  //   }
  // }
  // auto lambda_phi = autotensornormal.transpose()*lambda;
  VectorX<AutoDiffXd> lambda_phi =
      tensornormal.cast<AutoDiffXd>().transpose()*lambda;

  VectorX<AutoDiffXd> y1 = lambda_phi;

  if (debug) {
    using namespace std;
    cout << "lambda_phi: " << lambda_phi.transpose() << "\n";
  }
  VectorX<AutoDiffXd> y2 = phi;

  VectorX<AutoDiffXd> y3 = phi.transpose()*lambda_phi;

  y << y1, y2, y3;
}

TimestepIntegrationConstraint::TimestepIntegrationConstraint(
    const RigidBodyTree<double>& tree,
    const RigidBodyTree<double>& empty_tree,
    std::shared_ptr<plants::KinematicsCacheWithVHelper<AutoDiffXd>>
        kinematics_cache_with_v_helper, int num_lambda, double elasticity) // constraints: q; v+; v-; lambda
    : solvers::Constraint(num_lambda/3 + 1, num_lambda+2*tree.get_num_velocities() + tree.get_num_positions(),
        Eigen::MatrixXd::Zero(num_lambda/3 + 1, 1),
        Eigen::MatrixXd::Zero(num_lambda/3 + 1, 1)), 
      tree_(&tree),
      empty_tree_(&empty_tree),
      num_positions_{tree.get_num_positions()},
      num_velocities_{2*tree.get_num_velocities()},
      num_lambda_{num_lambda},
      num_contacts_{num_lambda/3},
      elasticity_{elasticity},
      kinematics_cache_with_v_helper_{kinematics_cache_with_v_helper} {
  unused(num_contacts_);
}

void TimestepIntegrationConstraint::DoEval(
  const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const {
  AutoDiffVecXd y_t;
  DoEval(math::initializeAutoDiff(x), y_t);
  y = math::autoDiffToValueMatrix(y_t);
}

void TimestepIntegrationConstraint::DoEval(
  const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const {

  int x_count = 0;

  auto x_segment = [x, &x_count](int num_element) {
    x_count += num_element;
    return x.segment(x_count - num_element, num_element);
  };

  const AutoDiffVecXd q = x_segment(num_positions_);
  const AutoDiffVecXd v_plus = x_segment(num_velocities_/2);
  const AutoDiffVecXd v_minus = x_segment(num_velocities_/2);
  const AutoDiffVecXd lambda = x_segment(num_lambda_);

  auto kinsol = kinematics_cache_with_v_helper_->UpdateKinematics(q, v_minus);

  const auto M = tree_->massMatrix(kinsol);

  const MatrixX<AutoDiffXd> map_v_to_qdot =
      RigidBodyTree<double>::GetVelocityToQDotMapping(kinsol);
  
  const AutoDiffVecXd qd_plus = map_v_to_qdot * v_plus;
  const AutoDiffVecXd qd_minus = map_v_to_qdot * v_minus;

  VectorX<AutoDiffXd> phi;
  MatrixX<AutoDiffXd> Jphi;
  VectorX<AutoDiffXd> lambda_calc;
  MatrixX<double> tensornormal;
  CollisionStuff(
      *tree_, *empty_tree_, q, v_minus, lambda,
      x.size(), elasticity_,
      &phi, &Jphi, &lambda_calc, &tensornormal);

  auto lambda_phi = tensornormal.cast<AutoDiffXd>().transpose()*lambda;
  auto dphi_delta = Jphi*qd_plus + elasticity_*Jphi*qd_minus;

  // Jqdot_plus_l - Jqdot_minus_l = -(1+restitution)J*M^(-1)*J^T*lambda
  y << Jphi*qd_plus - Jphi*qd_minus -
          Jphi*M.inverse()*Jphi.transpose()*lambda_phi,
          dphi_delta.transpose()*lambda_phi;
  //auto dphi_delta = Jphi*qd_plus + elasticity_*Jphi*qd_minus;
  //dphi_delta = 
}


}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake
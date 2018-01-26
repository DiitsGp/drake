#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/polynomial.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/mathematical_program.h"

using Eigen::Dynamic;
using Eigen::Ref;
using Eigen::Matrix;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

using drake::Vector1d;
using drake::solvers::detail::VecIn;
using drake::solvers::detail::VecOut;

using std::endl;
using std::make_shared;
using std::map;
using std::numeric_limits;
using std::ostringstream;
using std::pair;
using std::runtime_error;
using std::set;
using std::shared_ptr;
using std::static_pointer_cast;
using std::string;
using std::unique_ptr;
using std::vector;

namespace drake {
namespace solvers {
namespace test {

class EvalFunc1 : public EvaluatorBase {
public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EvalFunc1)
  EvalFunc1() : EvaluatorBase(2, 2) {}

protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const override {
  	y(0) = x(0)*x(0);
  	y(1) = x(1)*x(1);
  }
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd& y) const override {
  	y(0) = x(0)*x(0);
  	y(1) = x(1)*x(1);
  }
};


GTEST_TEST(testConstraint, testNLCC_Eval) {
  const VectorXd ub = VectorXd::Constant(1, 25.0);
  VectorXd x(4), x1(2), x2(2), y(1);
  x1 << 1, 1;
  x2 << 2, 2;
  x << x1, x2;
  auto f1 = std::make_shared<EvalFunc1>();
  auto f2 = std::make_shared<EvalFunc1>();

  NonlinearComplementarityConstraint nlcc(f1, f2, ub);
  nlcc.Eval(x, y);
  std::cout<<"The final output is: " << y << std::endl;
  std::cout<<"The expected output is: " << 8;
}


}  // namespace test
}  // namespace solvers
}  // namespace drake

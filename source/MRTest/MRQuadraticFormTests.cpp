#include <MRMesh/MRQuadraticForm.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRVector3.h>

namespace MR
{

// verifies that template can be instantiated with typical parameters
template struct QuadraticForm<Vector2<float>>;
template struct QuadraticForm<Vector2<double>>;
template struct QuadraticForm<Vector3<float>>;
template struct QuadraticForm<Vector3<double>>;

} //namespace MR

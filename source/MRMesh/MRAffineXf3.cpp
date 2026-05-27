#include "MRAffineXf3.h"

namespace MR
{

// verifies that template can be instantiated with typical parameters
template struct AffineXf<Vector3<float>>;
template struct AffineXf<Vector3<double>>;

} //namespace MR

#include "MRAffineXf2.h"

namespace MR
{

// verifies that template can be instantiated with typical parameters
template struct AffineXf<Vector2<float>>;
template struct AffineXf<Vector2<double>>;

} //namespace MR

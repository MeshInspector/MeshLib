#include "MRMatrix2.h"
#include "MRMatrix3.h"
#include "MRMatrix4.h"
#include "MRLine3.h"
#include "MRLineSegm3.h"
#include "MRPlane3.h"
#include "MRQuaternion.h"
#include "MRSphere.h"
#include "MRSymMatrix2.h"
#include "MRSymMatrix3.h"
#include "MRSymMatrix4.h"
#include "MRTriPoint.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRRigidScaleXf3.h"
#include <cmath>
#include <cassert>

namespace MR
{

// verifies that templates can be instantiated with typical parameters

template struct Matrix2<float>;
template struct Matrix2<double>;

template struct Matrix3<float>;
template struct Matrix3<double>;

template struct Matrix4<float>;
template struct Matrix4<double>;

template struct Sphere<Vector2f>;
template struct Sphere<Vector2d>;

template struct Sphere<Vector3f>;
template struct Sphere<Vector3d>;

template struct Line<Vector2f>;
template struct Line<Vector2d>;

template struct Line<Vector3f>;
template struct Line<Vector3d>;

template struct LineSegm<Vector2f>;
template struct LineSegm<Vector2d>;

template struct LineSegm<Vector3f>;
template struct LineSegm<Vector3d>;

template struct Plane3<float>;
template struct Plane3<double>;

template struct Quaternion<float>;
template struct Quaternion<double>;

template struct SymMatrix2<float>;
template struct SymMatrix2<double>;

template struct SymMatrix3<float>;
template struct SymMatrix3<double>;

template struct SymMatrix4<float>;
template struct SymMatrix4<double>;

template struct TriPoint<float>;
template struct TriPoint<double>;

template struct Vector2<float>;
template struct Vector2<double>;

template struct Vector3<float>;
template struct Vector3<double>;

template struct Vector4<float>;
template struct Vector4<double>;

template struct RigidXf3<float>;
template struct RigidXf3<double>;

template struct RigidScaleXf3<float>;
template struct RigidScaleXf3<double>;

template Matrix3<float>  slerp<float>(  const Matrix3<float> & m0,  const Matrix3<float> & m1,  float t );
template Matrix3<double> slerp<double>( const Matrix3<double> & m0, const Matrix3<double> & m1, double t );

template AffineXf3<float>  slerp( const AffineXf3<float> & xf0,  const AffineXf3<float> & xf1,  float t,  const Vector3<float> & p );
template AffineXf3<double> slerp( const AffineXf3<double> & xf0, const AffineXf3<double> & xf1, double t, const Vector3<double> & p );

template Matrix3<float>  orthonormalized( const Matrix3<float> & m );
template Matrix3<double> orthonormalized( const Matrix3<double> & m );

template AffineXf3<float>  orthonormalized( const AffineXf3<float> & xf,  const Vector3<float> & center );
template AffineXf3<double> orthonormalized( const AffineXf3<double> & xf, const Vector3<double> & center );

} //namespace MR

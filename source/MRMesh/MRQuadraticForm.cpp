#pragma warning(disable:4464)  //relative include path contains '..'
#pragma warning(disable:5054)  //operator '&': deprecated between enumerations of different types
#include "MRQuadraticForm.h"
#include "MRToFromEigen.h"

// unknown pragmas
#pragma warning(disable:4068)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#include <Eigen/Dense>
#pragma clang diagnostic pop

namespace MR
{

template <typename V>
std::pair< QuadraticForm<V>, V > sum(
    const QuadraticForm<V> & q0, const V & x0,
    const QuadraticForm<V> & q1, const V & x1,
    bool minAmong01 )
{
    using T = typename QuadraticForm<V>::T;
    QuadraticForm<V> q2;
    V x2;

    q2.A = q0.A + q1.A;

    if ( minAmong01 )
    {
        const auto c0 = q0.c + q1.eval( x0 - x1 );
        const auto c1 = q0.eval( x0 - x1 ) + q1.c;
        if ( c0 <= c1 )
        {
            q2.c = c0;
            x2 = x0;
        }
        else
        {
            q2.c = c1;
            x2 = x1;
        }
    }
    else
    {
        const auto center = T(0.5) * ( x0 + x1 );
        Eigen::Map<Eigen::Matrix<T, V::elements, 1>>{ &x2.x } = 
            toEigen( q2.A ).ldlt().solve( toEigen( q0.A * ( x0 - center ) + q1.A * ( x1 - center ) ) );
        x2 += center;
        q2.c = q0.eval( x0 - x2 ) + q1.eval( x1 - x2 );
    }

    return { q2, x2 };
}

// verifies that template can be instantiated with typical parameters
template struct QuadraticForm<Vector2<float>>;
template struct QuadraticForm<Vector2<double>>;
template struct QuadraticForm<Vector3<float>>;
template struct QuadraticForm<Vector3<double>>;

template MRMESH_API std::pair< QuadraticForm<Vector2<float>>,  Vector2<float> >  sum( const QuadraticForm<Vector2<float>> & q0,  const Vector2<float> & x0,  const QuadraticForm<Vector2<float>> & q1,  const Vector2<float> & x1, bool minAmong01 );
template MRMESH_API std::pair< QuadraticForm<Vector2<double>>, Vector2<double> > sum( const QuadraticForm<Vector2<double>> & q0, const Vector2<double> & x0, const QuadraticForm<Vector2<double>> & q1, const Vector2<double> & x1, bool minAmong01 );
template MRMESH_API std::pair< QuadraticForm<Vector3<float>>,  Vector3<float> >  sum( const QuadraticForm<Vector3<float>> & q0,  const Vector3<float> & x0,  const QuadraticForm<Vector3<float>> & q1,  const Vector3<float> & x1, bool minAmong01 );
template MRMESH_API std::pair< QuadraticForm<Vector3<double>>, Vector3<double> > sum( const QuadraticForm<Vector3<double>> & q0, const Vector3<double> & x0, const QuadraticForm<Vector3<double>> & q1, const Vector3<double> & x1, bool minAmong01 );

} //namespace MR

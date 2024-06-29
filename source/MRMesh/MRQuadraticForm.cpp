#include "MRQuadraticForm.h"

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
        x2 = q2.A.pseudoinverse() * ( q0.A * ( x0 - center ) + q1.A * ( x1 - center ) ) + center;
        q2.c = q0.eval( x0 - x2 ) + q1.eval( x1 - x2 );
    }

    return { q2, x2 };
}

template <typename V>
QuadraticForm<V> sumAt(
    const QuadraticForm<V> & q0, const V & x0,
    const QuadraticForm<V> & q1, const V & x1,
    const V & pos )
{
    QuadraticForm<V> q2;
    q2.A = q0.A + q1.A;
    q2.c = q0.eval( x0 - pos ) + q1.eval( x1 - pos );
    return q2;
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

template MRMESH_API QuadraticForm<Vector2<float>>  sumAt( const QuadraticForm<Vector2<float>> & q0,  const Vector2<float> & x0,  const QuadraticForm<Vector2<float>> & q1,  const Vector2<float> & x1, const Vector2<float> & pos );
template MRMESH_API QuadraticForm<Vector2<double>> sumAt( const QuadraticForm<Vector2<double>> & q0, const Vector2<double> & x0, const QuadraticForm<Vector2<double>> & q1, const Vector2<double> & x1, const Vector2<double> & pos );
template MRMESH_API QuadraticForm<Vector3<float>>  sumAt( const QuadraticForm<Vector3<float>> & q0,  const Vector3<float> & x0,  const QuadraticForm<Vector3<float>> & q1,  const Vector3<float> & x1, const Vector3<float> & pos );
template MRMESH_API QuadraticForm<Vector3<double>> sumAt( const QuadraticForm<Vector3<double>> & q0, const Vector3<double> & x0, const QuadraticForm<Vector3<double>> & q1, const Vector3<double> & x1, const Vector3<double> & pos );

} //namespace MR

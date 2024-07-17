#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

/// affine transformation: y = A*x + b, where A in VxV, and b in V
public ref class AffineXf3f
{
public:
    /// creates identity transformation
    AffineXf3f();
    /// creates linear-only transformation (without translation)
    AffineXf3f( Matrix3f^ A );
    /// creates translation-only transformation (with identity linear component)
    AffineXf3f( Vector3f^ b );
    /// creates full transformation
    AffineXf3f( Matrix3f^ A, Vector3f^ b );
    ~AffineXf3f();

    /// composition of two transformations:
    /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
    static AffineXf3f^ operator*( AffineXf3f^ a, AffineXf3f^ b );

    /// linear component
    property Matrix3f^ A { Matrix3f^ get(); void set( Matrix3f^ value ); }

    /// translation
    property Vector3f^ B { Vector3f^ get(); void set( Vector3f^ value ); }

private:
    Matrix3f^ A_;
    Vector3f^ b_;
    MR::AffineXf3f* xf_;

internal:
    AffineXf3f( MR::AffineXf3f* xf );
    MR::AffineXf3f* xf() { return xf_; }
};

MR_DOTNET_NAMESPACE_END


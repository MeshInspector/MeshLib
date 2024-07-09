#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

ref class Vector3f;
ref class Matrix3f;

public ref class AffineXf3f
{
public:
    AffineXf3f();
    AffineXf3f( Matrix3f^ A );
    AffineXf3f( Vector3f^ b );
    AffineXf3f( Matrix3f^ A, Vector3f^ b );
    ~AffineXf3f();

    static AffineXf3f^ operator*( AffineXf3f^ a, AffineXf3f^ b );

    property Matrix3f^ A { Matrix3f^ get(); void set( Matrix3f^ value ); }
    property Vector3f^ b { Vector3f^ get(); void set( Vector3f^ value ); }

private:
    Matrix3f^ A_;
    Vector3f^ b_;
    MR::AffineXf3f* xf_;

internal:
    AffineXf3f( MR::AffineXf3f* xf );
    MR::AffineXf3f* xf() { return xf_; }
};

MR_DOTNET_NAMESPACE_END


#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

ref class Vector3f;
public ref class Matrix3f
{
public:
    Matrix3f();
    Matrix3f( Vector3f^ x, Vector3f^ y, Vector3f^ z );
    ~Matrix3f();

    static Matrix3f^ Zero();

    static Matrix3f^ Rotation( Vector3f^ axis, float angle );
    static Matrix3f^ Rotation( Vector3f^ from, Vector3f^ to );


    property Vector3f^ X { Vector3f^ get(); void set( Vector3f^ value ); }
    property Vector3f^ Y { Vector3f^ get(); void set( Vector3f^ value ); }
    property Vector3f^ Z { Vector3f^ get(); void set( Vector3f^ value ); }

    static Matrix3f^ operator*( Matrix3f^ a, Matrix3f^ b );

    static Vector3f^ operator*( Matrix3f^ a, Vector3f^ b );

    static bool operator==( Matrix3f^ a, Matrix3f^ b );
    static bool operator!=( Matrix3f^ a, Matrix3f^ b );

internal:
    Matrix3f( MR::Matrix3f* mat );
    MR::Matrix3f* mat() { return mat_; }

private:
    MR::Matrix3f* mat_;   

    Vector3f^ x_;
    Vector3f^ y_;
    Vector3f^ z_;
};

MR_DOTNET_NAMESPACE_END

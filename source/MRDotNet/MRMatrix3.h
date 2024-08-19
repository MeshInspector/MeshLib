#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

/// arbitrary 3x3 matrix
public ref class Matrix3f
{
public:
    /// creates the identity matrix
    Matrix3f();
    /// creates matrix with given rows
    Matrix3f( Vector3f^ x, Vector3f^ y, Vector3f^ z );
    ~Matrix3f();

    /// creates zero matrix
    static Matrix3f^ Zero();
    /// creates rotation matrix around given axis with given angle
    static Matrix3f^ Rotation( Vector3f^ axis, float angle );
    /// creates rotation matrix from one vector to another
    static Matrix3f^ Rotation( Vector3f^ from, Vector3f^ to );

    /// first row
    property Vector3f^ X { Vector3f^ get(); void set( Vector3f^ value ); }
    /// second row
    property Vector3f^ Y { Vector3f^ get(); void set( Vector3f^ value ); }
    /// third row
    property Vector3f^ Z { Vector3f^ get(); void set( Vector3f^ value ); }

    
    static Matrix3f^ operator*( Matrix3f^ a, Matrix3f^ b );
    static Vector3f^ operator*( Matrix3f^ a, Vector3f^ b );

    static bool operator==( Matrix3f^ a, Matrix3f^ b );
    static bool operator!=( Matrix3f^ a, Matrix3f^ b );

    static Matrix3f^ operator-( Matrix3f^ a, Matrix3f^ b );
    static Matrix3f^ operator+( Matrix3f^ a, Matrix3f^ b );

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

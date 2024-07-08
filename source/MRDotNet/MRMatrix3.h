#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

ref class Vector3f;
public ref class Matrix3f
{
    Matrix3f();
    Matrix3f( Vector3f^ x, Vector3f^ y, Vector3f^ z );
    ~Matrix3f();

    static Matrix3f^ zero();

    static Matrix3f^ rotation( Vector3f^ axis, float angle );
    static Matrix3f^ rotation( Vector3f^ from, Vector3f^ to );


    property Vector3f^ x { Vector3f^ get(); void set( Vector3f^ value ); }
    property Vector3f^ y { Vector3f^ get(); void set( Vector3f^ value ); }
    property Vector3f^ z { Vector3f^ get(); void set( Vector3f^ value ); }

    static Matrix3f^ operator*( Matrix3f^ a, Matrix3f^ b );

private:
    MR::Matrix3f* mat_;
    Matrix3f( MR::Matrix3f* mat );
};

MR_DOTNET_NAMESPACE_END

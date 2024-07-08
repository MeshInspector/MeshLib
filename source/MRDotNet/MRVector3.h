#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class Vector3f
{
public:
    Vector3f();
    Vector3f( float x, float y, float z );
    ~Vector3f();

    static Vector3f^ diagonal( float a );
    static Vector3f^ plusX();
    static Vector3f^ minusX();
    static Vector3f^ plusY();
    static Vector3f^ minusY();
    static Vector3f^ plusZ();
    static Vector3f^ minusZ();

    static Vector3f^ operator+( Vector3f^ a, Vector3f^ b );
    static Vector3f^ operator-( Vector3f^ a, Vector3f^ b );
    static Vector3f^ operator*( Vector3f^ a, float b );
    static Vector3f^ operator*( float a, Vector3f^ b );

    property float x { float get(); void set( float value ); }
    property float y { float get(); void set( float value ); }
    property float z { float get(); void set( float value ); }    

private:
    MR::Vector3f* vec_;

internal:
    Vector3f( MR::Vector3f* vec );
    MR::Vector3f* vec() { return vec_; }
};

MR_DOTNET_NAMESPACE_END


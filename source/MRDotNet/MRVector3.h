#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class Vector3f
{
public:
    Vector3f();
    Vector3f( float x, float y, float z );
    ~Vector3f();

    static Vector3f^ Diagonal( float a );
    static Vector3f^ PlusX();
    static Vector3f^ MinusX();
    static Vector3f^ PlusY();
    static Vector3f^ MinusY();
    static Vector3f^ PlusZ();
    static Vector3f^ MinusZ();

    static Vector3f^ operator+( Vector3f^ a, Vector3f^ b );
    static Vector3f^ operator-( Vector3f^ a, Vector3f^ b );
    static Vector3f^ operator*( Vector3f^ a, float b );
    static Vector3f^ operator*( float a, Vector3f^ b );

    property float X { float get(); void set( float value ); }
    property float Y { float get(); void set( float value ); }
    property float Z { float get(); void set( float value ); }

    static bool operator == ( Vector3f^ a, Vector3f^ b );
    static bool operator != ( Vector3f^ a, Vector3f^ b );    

private:
    MR::Vector3f* vec_;

internal:
    Vector3f( MR::Vector3f* vec );
    MR::Vector3f* vec() { return vec_; }
};

MR_DOTNET_NAMESPACE_END


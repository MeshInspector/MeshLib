#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

/// represents a 3-dimentional float-typed vector
public ref class Vector3f
{
public:
    /// creates a new vector with zero coordinates
    Vector3f();
    /// creates a new vector with specified coordinates
    Vector3f( float x, float y, float z );
    ~Vector3f();

    /// creates a new vector with same coordinates
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

    /// first coordinate
    property float X { float get(); void set( float value ); }
    /// second coordinate
    property float Y { float get(); void set( float value ); }
    /// third coordinate
    property float Z { float get(); void set( float value ); }

    static bool operator == ( Vector3f^ a, Vector3f^ b );
    static bool operator != ( Vector3f^ a, Vector3f^ b );    

private:
    MR::Vector3<float>* vec_;

internal:
    Vector3f( MR::Vector3<float>* vec );
    MR::Vector3<float>* vec() { return vec_; }
};

MR_DOTNET_NAMESPACE_END


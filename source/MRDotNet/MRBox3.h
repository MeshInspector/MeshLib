#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class Box3f
{
public:
    Box3f();    
    Box3f( Vector3f^ min, Vector3f^ max );    
    ~Box3f();

    static Box3f^ fromMinAndSize( Vector3f^ min, Vector3f^ size );

    property Vector3f^ Min { Vector3f^ get(); void set( Vector3f^ value ); }
    property Vector3f^ Max { Vector3f^ get(); void set( Vector3f^ value ); }

    Vector3f^ Center();
    Vector3f^ Size();
    float Diagonal();
    float Volume();

private:
    MR::Box3f* box_;

    Vector3f^ min_;
    Vector3f^ max_;

internal:
    Box3f( MR::Box3f* box );
    MR::Box3f* box() { return box_; }
};

MR_DOTNET_NAMESPACE_END


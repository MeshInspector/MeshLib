#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

/// Box given by its min- and max- corners
public ref class Box3f
{
public:
    /// create invalid box by default
    Box3f();
    /// create box with given corners
    Box3f( Vector3f^ min, Vector3f^ max );    
    ~Box3f();

    /// create box from min-corner and size
    static Box3f^ fromMinAndSize( Vector3f^ min, Vector3f^ size );

    property Vector3f^ Min { Vector3f^ get(); void set( Vector3f^ value ); }
    property Vector3f^ Max { Vector3f^ get(); void set( Vector3f^ value ); }
    
    /// computes center of the box
    Vector3f^ Center();
    /// computes size of the box in all dimensions
    Vector3f^ Size();
    /// computes length from min to max
    float Diagonal();
    /// computes the volume of this box
    float Volume();
    /// true if the box contains at least one point
    bool Valid();

private:
    MR::Box3f* box_;

    Vector3f^ min_;
    Vector3f^ max_;

internal:
    Box3f( MR::Box3f* box );
    MR::Box3f* box() { return box_; }
};

MR_DOTNET_NAMESPACE_END


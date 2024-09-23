#pragma once

#include "MRVoxelsFwd.h"

#include "MRMesh/MRExpected.h"
#include "MRVoxelsVolume.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRphmap.h"

#include <array>
#include <optional>

namespace MR
{

/// This class represents tooth id
class DentalId
{
public:
    /// Creates id from FDI number known at compile time
    template <int id>
    static constexpr DentalId fromFDI()
    {
        const int t = id % 10;
        const int q = id / 10;
        static_assert( q >= 1 && q <= 4 && t >= 1 && t <= 8 );

        return DentalId( id );
    }

    /// Creates id from FDI number known only at runtime
    MRVOXELS_API static std::optional<DentalId> fromFDI( int id );

    /// Returns FDI representation of the id
    MRVOXELS_API int fdi() const;

    auto operator <=> ( const DentalId& other ) const = default;

private:
    constexpr explicit DentalId( int fdi ):
        fdi_( fdi )
    {}

    int fdi_;
};

}

template <>
struct std::hash<MR::DentalId>
{
    inline size_t operator() ( const MR::DentalId& id ) const noexcept
    {
        return hash<int>{}( id.fdi() );
    }
};

namespace MR
{

/// This class is an alternative to directly invoking \ref meshToDirectionVolume for the mesh retrieved from the teeth mask.
/// It is better because when a single mesh is created from mask, some neighboring teeth might fuse together, creating incorrect mask.
/// This class invokes meshing for each teeth separately, thus eliminating this problem.
class TeethMaskToDirectionVolumeConvertor
{
public:
    /// Initialize class
    /// @param volume Voxel mask
    /// @param additionalIds List of additional ids (besides teeth) to convert
    MRVOXELS_API static Expected<TeethMaskToDirectionVolumeConvertor> create( const VdbVolume& volume, const std::vector<int>& additionalIds = {} );

    /// Returns all the objects present in volume and corresponding bounding boxes
    MRVOXELS_API const HashMap<int, Box3i>& getObjectBounds() const;

    /// See \ref meshToDirectionVolume for details
    using DirectionVolume = std::array<SimpleVolumeMinMax, 3>;
    struct ProcessResult
    {
        DirectionVolume volume;
        AffineXf3f xf;
    };

    /// Converts single object into direction volume
    MRVOXELS_API Expected<ProcessResult> convertObject( int id ) const;

    /// Converts all the objects into direction volume
    MRVOXELS_API Expected<ProcessResult> convertAll() const;

private:
    MRVOXELS_API TeethMaskToDirectionVolumeConvertor();

    HashMap<int, Box3i> presentObjects_;
    SimpleVolume mask_;
};


/// A shortcut for \ref TeethMaskToDirectionVolumeConvertor::create and \ref TeethMaskToDirectionVolumeConvertor::convertAll
MRVOXELS_API Expected<std::array<SimpleVolumeMinMax, 3>> teethMaskToDirectionVolume( const VdbVolume& volume, const std::vector<int>& additionalIds = {} );


}

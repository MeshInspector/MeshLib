#pragma once

#include "MRPrecisePredicates3.h"
#include "MRIntersectionContour.h"
#include "MRExpected.h"
#include "MREnums.h"
#include "MRBitSet.h"
#include <span>

namespace MR
{

// Special data to sort intersections more accurate
struct SortIntersectionsData
{
    const Mesh& otherMesh;
    const ContinuousContours& contours;
    ConvertToIntVector converter;
    const AffineXf3f* rigidB2A{nullptr};
    size_t meshAVertsNum;
    bool isOtherA{false};
};

// Simple point on mesh, represented by primitive id and coordinate in mesh space
struct OneMeshIntersection
{
    /// the kind of mesh primitive containing the point
    enum class Primitive : unsigned char
    {
        Face,
        Edge,
        Vert
    };

    OneMeshIntersection() noexcept : id_( -1 ), primitive_( Primitive::Face ) {}
    explicit OneMeshIntersection( NoInit ) noexcept : coordinate( noInit ) {}
    OneMeshIntersection( FaceId f, const Vector3f& p ) noexcept : coordinate( p ), id_( int( f ) ), primitive_( Primitive::Face ) {}
    OneMeshIntersection( EdgeId e, const Vector3f& p ) noexcept : coordinate( p ), id_( int( e ) ), primitive_( Primitive::Edge ) {}
    OneMeshIntersection( VertId v, const Vector3f& p ) noexcept : coordinate( p ), id_( int( v ) ), primitive_( Primitive::Vert ) {}

    [[nodiscard]] Primitive primitive() const { return primitive_; }
    [[nodiscard]] bool isFace() const { return primitive_ == Primitive::Face; }
    [[nodiscard]] bool isEdge() const { return primitive_ == Primitive::Edge; }
    [[nodiscard]] bool isVert() const { return primitive_ == Primitive::Vert; }

    /// the primitive containing the point if it is of that kind, otherwise invalid id
    [[nodiscard]] FaceId face() const { return isFace() ? FaceId( id_ ) : FaceId(); }
    [[nodiscard]] EdgeId edge() const { return isEdge() ? EdgeId( id_ ) : EdgeId(); }
    [[nodiscard]] VertId vert() const { return isVert() ? VertId( id_ ) : VertId(); }

    void setPrimitive( FaceId f ) { id_ = int( f ); primitive_ = Primitive::Face; }
    void setPrimitive( EdgeId e ) { id_ = int( e ); primitive_ = Primitive::Edge; }
    void setPrimitive( VertId v ) { id_ = int( v ); primitive_ = Primitive::Vert; }

    /// whether both points are in the same primitive
    [[nodiscard]] bool samePrimitive( const OneMeshIntersection& other ) const { return primitive_ == other.primitive_ && id_ == other.id_; }

    Vector3f coordinate;

private:
    int id_; // no default member initializers: the NoInit constructor must not write anything
    Primitive primitive_;
};

// One contour on mesh
struct OneMeshContour
{
    Vector<OneMeshIntersection> intersections;
    bool closed{false};
};

// One contour of OneMeshContours, viewed in place
struct MR_BIND_IGNORE OneMeshContourView
{
    std::span<const OneMeshIntersection> intersections;
    bool closed{false};
};

/** \ingroup BooleanGroup
  * \brief Special data type for MR::cutMesh: the intersections of all contours in one array, one contour after another
  *
  * This is special data for MR::cutMesh, you can convert some other contours representation to it by:\n
  * \ref MR::surfaceLineToClosedMeshContour
  * \ref MR::surfacePathsToMeshContours
  */
struct OneMeshContours
{
    Vector<OneMeshIntersection> intersections;
    Vector<int> offsets{ 0 }; ///< the number of contours + 1 elements: contour c is intersections[offsets[c], offsets[c + 1])
    BitSet closed;

    OneMeshContours() = default;
    OneMeshContours( std::initializer_list<OneMeshContour> contours ) { for ( const auto& c : contours ) push_back( c ); }
    MRMESH_API OneMeshContours( const Vector<OneMeshContour>& contours );

    [[nodiscard]] size_t size() const { return offsets.size() - 1; }
    [[nodiscard]] bool empty() const { return offsets.size() == 1; }
    [[nodiscard]] MR_BIND_IGNORE OneMeshContourView operator[]( size_t c ) const
        { return { { intersections.data() + offsets[c], intersections.data() + offsets[c + 1] }, closed.test( c ) }; }
    [[nodiscard]] const OneMeshIntersection& intersection( size_t c, size_t i ) const { return intersections[offsets[c] + i]; }

    /// iterates over the views of the contours
    struct MR_BIND_IGNORE ConstIterator
    {
        const OneMeshContours* contours = nullptr;
        size_t c = 0;
        [[nodiscard]] OneMeshContourView operator*() const { return ( *contours )[c]; }
        ConstIterator& operator++() { ++c; return *this; }
        [[nodiscard]] bool operator==( const ConstIterator& ) const = default;
    };
    [[nodiscard]] MR_BIND_IGNORE ConstIterator begin() const { return { this, 0 }; }
    [[nodiscard]] MR_BIND_IGNORE ConstIterator end() const { return { this, size() }; }

    /// returns a copy of contour c
    [[nodiscard]] MRMESH_API OneMeshContour contour( size_t c ) const;

    /// appends a contour
    MR_BIND_IGNORE MRMESH_API void push_back( const OneMeshContourView& contour );
    void push_back( const OneMeshContour& contour ) { push_back( { { contour.intersections.data(), contour.intersections.size() }, contour.closed } ); }
};

/// calls f( c, i0, i1 ) in parallel on pieces [i0, i1) of all elements of contours with given offsets, every piece within contour c
MR_BIND_IGNORE MRMESH_API void parallelForContourPieces( const Vector<int>& offsets, const std::function<void( int c, int i0, int i1 )>& f );

// Divides faces that fully own contours into 3 parts with center in center mass of one of the face contours
// if there is more than one contour on face it guarantee to subdivide at least one lone contour on this face
// the AABB tree of the mesh (if any) is updated instead of being invalidated
[[deprecated( "cutMesh connects lone contours itself, and the boolean no longer uses this" )]]
MRMESH_API void subdivideLoneContours( Mesh& mesh, const OneMeshContours& contours, FaceHashMap* new2oldMap = nullptr );

/// Converts contours given in topological terms as the intersections of one mesh's edge and another mesh's triangle (ContinuousContours),
/// into contours of meshA and/or meshB given as a sequence of (primitive id and Cartesian coordinates);
/// converters are required for better precision in case of degenerations;
/// note that contours should not have intersections
MRMESH_API void getOneMeshIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& contours,
    OneMeshContours* outA, OneMeshContours* outB,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A = nullptr,
    Contours3f* outPtsA = nullptr,
    bool addSelfyTerminalVerts = false ); ///< if true, then open self-intersection contours will be prolonged to terminal vertices

// Converts ordered continuous self contours of single meshes to OneMeshContours
// converters are required for better precision in case of degenerations
[[nodiscard]]
MRMESH_API OneMeshContours getOneMeshSelfIntersectionContours( const Mesh& mesh, const ContinuousContours& contours,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A = nullptr );

// Converts OneMeshContours contours representation to Contours3f: set of coordinates
[[nodiscard]]
MRMESH_API Contours3f extractMeshContours( const OneMeshContours& meshContours );

using MeshTriPointsConnector = std::function<Expected<SurfacePath>( const MeshTriPoint& start, const MeshTriPoint& end, int startIndex, int endIndex )>;
/** \ingroup BooleanGroup
  * \brief Makes continuous contour by mesh tri points, if first and last meshTriPoint is the same, makes closed contour
  *
  * Finds paths between neighbor \p surfaceLine with MeshTriPointsConnector function and build contour MR::cutMesh input
  * \param connectorFn function to build path between neighbor surfaceLine, if not present simple geodesic path function is used
  * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
  */
[[nodiscard]]
MR_BIND_IGNORE MRMESH_API Expected<OneMeshContour> surfaceLineToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    MeshTriPointsConnector connectorFn, std::vector<int>* pivotIndices = nullptr );

/// Geo path search settings
struct SearchPathSettings
{
    GeodesicPathApprox geodesicPathApprox{ GeodesicPathApprox::DijkstraAStar }; ///< the algorithm to compute approximately geodesic path
    int maxReduceIters{ 100 }; ///< the maximum number of iterations to reduce approximate path length and convert it in geodesic path
};

/** \ingroup BooleanGroup
  * \brief Makes continuous contour by mesh tri points, if first and last meshTriPoint is the same, makes closed contour
  *
  * Finds shortest paths between neighbor \p surfaceLine and build contour MR::cutMesh input
  * \param searchSettings settings for search geo path 
  * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
  */
[[nodiscard]]
MRMESH_API Expected<OneMeshContour> surfaceLineToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr );

/** \ingroup BooleanGroup
  * \brief Makes closed continuous contour by mesh tri points, note that first and last meshTriPoint should not be same
  * 
  * Finds shortest paths between neighbor \p surfaceLine and build closed contour MR::cutMesh input
  * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
  * \note better use surfaceLineToMeshContour(...) instead, note that it requires same front and back MeshTriPoints for closed contour
  */
[[nodiscard]]
MRMESH_API Expected<OneMeshContour> surfaceLineToClosedMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr );

/** \ingroup BooleanGroup
  * \brief Converts SurfacePath to OneMeshContours
  *
  * Creates MR::OneMeshContour object from given surface path with ends for MR::cutMesh input
  * `start` and surfacePath.front() should be from same face
  * surfacePath.back() and `end` should be from same face
  * 
  * note that whole path (including `start` and `end`) should not have self-intersections
  * also following case is not supported (vertex -> edge (incident with vertex)):
  * 
  * vert path  edge point path     edge end
  * o----------o-  --  --  --  --  O
  *  \          \                /
  *       \      \          /
  *            \  \     /
  *               \\/
  *                 o path
  */
[[nodiscard]]
MRMESH_API OneMeshContour surfacePathWithEndsToMeshContour( const Mesh& mesh, 
                                                                          const MeshTriPoint& start, 
                                                                          const SurfacePath& surfacePath, 
                                                                          const MeshTriPoint& end );

/** \ingroup BooleanGroup
  * \brief Converts SurfacePaths to OneMeshContours
  * 
  * Creates MR::OneMeshContours object from given surface paths for MR::cutMesh input
  */
[[nodiscard]]
MRMESH_API OneMeshContours surfacePathsToMeshContours( const Mesh& mesh, const std::vector<SurfacePath>& surfacePaths );

// the old names stay in the bindings, since scripts call them
[[deprecated( "use surfaceLineToMeshContour" )]] [[nodiscard]]
MR_BIND_IGNORE inline Expected<OneMeshContour> convertMeshTriPointsToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    MeshTriPointsConnector connectorFn, std::vector<int>* pivotIndices = nullptr )
{
    return surfaceLineToMeshContour( mesh, surfaceLine, std::move( connectorFn ), pivotIndices );
}

[[deprecated( "use surfaceLineToMeshContour" )]] [[nodiscard]]
inline Expected<OneMeshContour> convertMeshTriPointsToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr )
{
    return surfaceLineToMeshContour( mesh, surfaceLine, searchSettings, pivotIndices );
}

[[deprecated( "use surfaceLineToClosedMeshContour" )]] [[nodiscard]]
inline Expected<OneMeshContour> convertMeshTriPointsToClosedContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr )
{
    return surfaceLineToClosedMeshContour( mesh, surfaceLine, searchSettings, pivotIndices );
}

[[deprecated( "use surfacePathWithEndsToMeshContour" )]] [[nodiscard]]
inline OneMeshContour convertSurfacePathWithEndsToMeshContour( const Mesh& mesh, const MeshTriPoint& start, const SurfacePath& surfacePath, const MeshTriPoint& end )
{
    return surfacePathWithEndsToMeshContour( mesh, start, surfacePath, end );
}

[[deprecated( "use surfacePathsToMeshContours" )]] [[nodiscard]]
inline OneMeshContours convertSurfacePathsToMeshContours( const Mesh& mesh, const std::vector<SurfacePath>& surfacePaths )
{
    return surfacePathsToMeshContours( mesh, surfacePaths );
}
} //namespace MR

#pragma once

#include "MRBitSet.h"
#include "MRVector.h"
#include "MRVector3.h"
#include "MRMeshTriPoint.h"
#include "MREnums.h"

namespace MR
{

/// Laplacian to smoothly deform a region preserving mesh fine details.
/// How to use:
/// 1. Initialize Laplacian for the region being deformed, here region properties are remembered.
/// 2. Change positions of some vertices within the region and call fixVertex for them.
/// 3. Optionally call updateSolver()
/// 4. Call apply() to change the remaining vertices within the region
/// Then steps 1-4 or 2-4 can be repeated.
/// \snippet cpp-samples/LaplacianDeformation.cpp 0
class Laplacian
{
public:
    MRMESH_API explicit Laplacian( Mesh & mesh );
    MRMESH_API Laplacian( const MeshTopology & topology, VertCoords & points );
    MRMESH_API Laplacian( Laplacian && ) noexcept;
    MRMESH_API ~Laplacian();

    /// (re)initialize Laplacian for the region being deformed, here region properties are remembered and precomputed;
    /// \param freeVerts must not include all vertices of a mesh connected component;
    /// \param usePoints is not nullptr, then these points will be used instead of passed to a constructor for weights computation and shape memory
    void init( const VertBitSet & freeVerts, EdgeWeights weights, VertexMass vmass = VertexMass::Unit, RememberShape rem = RememberShape::Yes )
        { initFromPoints( points_, freeVerts, weights, vmass, rem ); }

    /// same as init() but uses the given points instead of ones passed to a constructor for weights computation and shape memory
    MRMESH_API void initFromPoints( const VertCoords & points, const VertBitSet & freeVerts, EdgeWeights weights,
        VertexMass vmass = VertexMass::Unit, RememberShape rem = RememberShape::Yes );

    /// notify Laplacian that given vertex has changed after init and must be fixed during apply;
    /// \param smooth whether to make the surface smooth in this vertex (sharp otherwise)
    MRMESH_API void fixVertex( VertId v, bool smooth = true );

    /// sets position of given vertex after init and it must be fixed during apply (THIS METHOD CHANGES THE MESH);
    /// \param smooth whether to make the surface smooth in this vertex (sharp otherwise)
    MRMESH_API void fixVertex( VertId v, const Vector3f & fixedPos, bool smooth = true );

    /// multiplies vertex equation's weight on the given factor
    MRMESH_API void multVertexWeight( VertId v, double factor );

    /// if you manually call this method after initialization and fixing vertices then next apply call will be much faster
    MRMESH_API void updateSolver();

    /// takes fixed vertex positions from the given points vector,
    /// computes and writes free vertex positions in the given points vector as well
    MRMESH_API void applyToVector( VertCoords & points );

    /// takes fixed vertex positions from the points vector passed to a constructor,
    /// computes and writes free vertex positions in the points vector passed to a constructor as well
    void apply() { applyToVector( points_ ); }

    /// takes fixed vertex scalars from the given field,
    /// computes and writes free vertex scalars in the given field as well
    MRMESH_API void applyToScalar( VertScalars & scalarField );

    /// return all initially free vertices and the first layer of vertices around them
    [[nodiscard]] const VertBitSet & region() const { return region_; }

    /// return currently free vertices
    [[nodiscard]] const VertBitSet & freeVerts() const { return freeVerts_; }

    /// return fixed vertices from the first layer around free vertices
    [[nodiscard]] const VertBitSet & firstLayerFixedVerts() const { assert( solver_ ); return firstLayerFixedVerts_; }

    /// return the topology for which Laplacian was constructed
    [[nodiscard]] const MeshTopology & topology() const { return topology_; }

    /// return the vector of coordinates for which Laplacian was constructed
    [[nodiscard]] VertCoords & points() const { return points_; }

    /// attracts the given point inside some mesh's triangle to the given target with the given weight
    struct Attractor
    {
        MeshTriPoint p;
        Vector3d target;
        /// the weight or priority of this attractor relative to all other equations,
        /// which must be compatible with weights of other equations;
        /// the weight of ordinary equations is 1 for VertexMass::Unit,
        /// and 1 / sqrt( double area around central vertex ) for VertexMass::NeiArea
        double weight = 1;
    };

    /// adds one more attractor to the stored list
    MRMESH_API void addAttractor( const Attractor& a );

    /// forgets all attractors added previously
    MRMESH_API void removeAllAttractors();

private:
    template <typename I, typename G, typename S, typename P>
    void prepareRhs_( I && iniRhs, G && g, S && s, P && p ) const;

    const MeshTopology & topology_;
    VertCoords & points_;

    // all initially free vertices and the first layer of vertices around them
    VertBitSet region_;

    // currently free vertices
    VertBitSet freeVerts_;

    // fixed vertices where no smoothness is required
    VertBitSet fixedSharpVertices_;

    // fixed vertices from the first layer around free vertices
    VertBitSet firstLayerFixedVerts_;

    // for all vertices in the region
    struct Equation
    {
        Vector3d rhs;           // equation right hand side
        double centerCoeff = 0; // coefficient on matrix diagonal
        int firstElem = 0;      // index in nonZeroElements_
    };
    std::vector<Equation> equations_;

    std::vector<Attractor> attractors_;

    struct Element
    {
        double coeff = 0;
        VertId neiVert;
    };
    std::vector<Element> nonZeroElements_;

    // map from vertex index to matrix row/col
    Vector< int, VertId > regionVert2id_;
    Vector< int, VertId > freeVert2id_;

    class Solver;
    std::unique_ptr<Solver> solver_;
};

} //namespace MR

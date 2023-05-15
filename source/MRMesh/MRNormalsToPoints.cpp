#include "MRNormalsToPoints.h"
#include "MRMesh.h"
#include "MRTimer.h"

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wunknown-warning-option" // for next one
#pragma clang diagnostic ignored "-Wunused-but-set-variable" // for newer clang
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

namespace
{

class Solver : public NormalsToPoints::ISolver
{
public:
    virtual void prepare( const MeshTopology & topology ) override;

private:
    constexpr static double guideWeight = 0.001;
    using SparseMatrixColMajor = Eigen::SparseMatrix<double,Eigen::ColMajor>;
    Eigen::SimplicialLDLT<SparseMatrixColMajor> ldlt_;
};

void Solver::prepare( const MeshTopology & topology )
{
    MR_TIMER
    std::vector< Eigen::Triplet<double> > mTriplets;
    const int nVerts = (int)topology.vertSize();
    mTriplets.reserve( nVerts + 6 * topology.numValidFaces() );
    // every point shall be close to corresponding guide point (with small weight)
    for ( int v = 0; v < nVerts; ++v )
        mTriplets.emplace_back( v, v, guideWeight );

    // add 2 equations per triangle for relative position of triangle points
    const int nRows = nVerts + 2 * topology.numValidFaces();
    int row = nVerts;
    for ( auto f : topology.getValidFaces() )
    {
        VertId vs[3];
        topology.getTriVerts( f, vs );

        mTriplets.emplace_back( row, vs[0],  2 );
        mTriplets.emplace_back( row, vs[1], -1 );
        mTriplets.emplace_back( row, vs[2], -1 );
        ++row;

        mTriplets.emplace_back( row, vs[0], -1 );
        mTriplets.emplace_back( row, vs[1],  2 );
        mTriplets.emplace_back( row, vs[2], -1 );
        ++row;
    }
    assert( row == nRows );
    assert( mTriplets.size() == nVerts + 6 * topology.numValidFaces() );

    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SparseMatrix M;
    M.resize( nRows, nVerts );
    M.setFromTriplets( mTriplets.begin(), mTriplets.end() );

    SparseMatrix A = M.adjoint() * M;
    ldlt_.compute( A );
}

} //anonymous namespace

} //namespace MR

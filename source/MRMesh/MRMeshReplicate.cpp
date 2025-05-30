#include "MRMeshReplicate.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4464) // relative include path contains '..'
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wunknown-warning-option" // for next one
#pragma clang diagnostic ignored "-Wunused-but-set-variable" // for newer clang
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

void replicateZ( Mesh & m, const Mesh & target )
{
    MR_TIMER;

    const auto szM = m.topology.numValidVerts();
    const auto szT =  target.topology.numValidVerts();
    if ( szT < szM )
    {
        // target mesh shall have more vertices not to have underdetermined system of equations
        assert( false );
        return;
    }

    Vector<MeshTriPoint, VertId> targetVertProjections;
    targetVertProjections.resizeNoInit( target.topology.vertSize() );
    m.getAABBTree();
    BitSetParallelFor( target.topology.getValidVerts(), [&]( VertId v )
    {
        targetVertProjections[v] = findProjection( target.points[v], m ).mtp;
    } );

    Vector<int, VertId> mVertToNum = makeVectorWithSeqNums( m.topology.getValidVerts() );

    std::vector< Eigen::Triplet<double> > mTriplets;
    mTriplets.reserve( szT * 3 );
    Eigen::VectorXd rhs( szT );
    int n = 0;
    for ( VertId v : target.topology.getValidVerts() )
    {
        for ( const auto & wv : targetVertProjections[v].getWeightedVerts( m.topology ) )
            mTriplets.emplace_back( n, mVertToNum[wv.v], wv.weight );
        rhs[ n++ ] = target.points[v].z;
    }
    assert( mTriplets.size() == szT * 3 );

    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SparseMatrix mat( szT, szM );
    mat.setFromTriplets( mTriplets.begin(), mTriplets.end() );

    SparseMatrix A = mat.adjoint() * mat;
    using SparseMatrixColMajor = Eigen::SparseMatrix<double,Eigen::ColMajor>;
    Eigen::SimplicialLDLT<SparseMatrixColMajor> ldlt;
    ldlt.compute( A );

    Eigen::VectorXd sol = ldlt.solve( mat.adjoint() * rhs );
    n = 0;
    for ( auto v : m.topology.getValidVerts() )
        m.points[v].z = float( sol[n++] );
}

} //namespace MR

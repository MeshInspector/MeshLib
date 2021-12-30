#include "MRLaplacian.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRExpandShrink.h"
#include "MRRingIterator.h"
#include "MRUVSphere.h"
#include "MRGTest.h"
#include <tbb/parallel_for.h>

// unknown pragmas
#pragma warning(disable:4068)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#include <Eigen/CholmodSupport>
#pragma clang diagnostic pop

namespace MR
{

void Laplacian::init( const VertBitSet & freeVerts, EdgeWeights weights, RememberShape rem )
{
    MR_TIMER;

    class CholmodeSolver final : public Solver
    {
    public:
        virtual void compute( const SparseMatrixColMajor& A ) final
        {
            solver_.compute( A );
        }

        virtual Eigen::VectorXd solve( const Eigen::VectorXd& rhs ) final
        {
            return solver_.solve( rhs );
        }
    private:
        Eigen::CholmodSimplicialLLT<SparseMatrixColMajor> solver_;
    };

    solver_ = std::make_unique<CholmodeSolver>();

    freeVerts_ = freeVerts;
    region_ = freeVerts;
    // free vertices and the first layer around the region
    expand( mesh_.topology, region_ );

    // build matrix of equations: vertex pos = mean pos of its neighbors
    regionVert2id_.resize( region_.size() );
    equations_.clear();
    nonZeroElements_.clear();
    std::vector<Element> rowElements;
    for ( auto v : region_ )
    {
        rowElements.clear();
        regionVert2id_[v] = (int)equations_.size();
        Equation eq;
        eq.firstElem = (int)nonZeroElements_.size();
        double sumW = 0;
        Vector3d sumWPos;
        for ( auto e : orgRing( mesh_.topology, v ) )
        {
            double w = 1;
            if ( weights == EdgeWeights::Cotan ) 
                w = mesh_.cotan( e );
            else if ( weights == EdgeWeights::CotanTimesLength ) 
                w = mesh_.edgeLength( e ) * mesh_.cotan( e );
            auto d = mesh_.topology.dest( e );
            rowElements.push_back( { -w, d } );
            sumWPos -= w * Vector3d( mesh_.points[d] );
            sumW += w;
        }
        const double rSumW = 1 / sumW;
        for ( auto el : rowElements )
        {
            el.coeff *= rSumW;
            nonZeroElements_.push_back( el );
        }
        eq.rhs = ( rem == RememberShape::Yes ) ? sumWPos * rSumW + Vector3d( mesh_.points[v] ) : Vector3d();
        eq.centerCoeff = 1;
        equations_.push_back( eq );
    }
    Equation eq;
    eq.firstElem = (int)nonZeroElements_.size();
    equations_.push_back( eq );
}

void Laplacian::fixVertex( VertId v ) 
{
    rhsValid_ = false;
    if ( freeVerts_.autoResizeTestSet( v, false ) )
        solverValid_ = false;
}

void Laplacian::fixVertex( VertId v, const Vector3f & fixedPos ) 
{ 
    mesh_.points[v] = fixedPos; 
    fixVertex( v ); 
}

void Laplacian::updateSolver()
{
    updateSolver_();
    updateRhs_();
}

void Laplacian::updateSolver_()
{
    if ( solverValid_ )
        return;
    solverValid_ = true;

    MR_TIMER

    const auto sz = freeVerts_.count();
    if ( sz <= 0 )
    {
        rhsValid_ = true;
        return;
    }
    rhsValid_ = false;

    freeVert2id_.resize( freeVerts_.size() );
    int n = 0;
    for ( auto v : freeVerts_ )
        freeVert2id_[v] = n++;

    firstLayerFixedVerts_ = freeVerts_;
    expand( mesh_.topology, firstLayerFixedVerts_ );
    const auto rowSz = firstLayerFixedVerts_.count();
    firstLayerFixedVerts_ -= freeVerts_;

    std::vector< Eigen::Triplet<double> > mTriplets;
    // equations for free vertices
    n = 0;
    for ( auto v : freeVerts_ )
    {
        assert( n == freeVert2id_[v] );
        int eqN = regionVert2id_[v];
        const auto eq = equations_[eqN];
        mTriplets.emplace_back( n, n, eq.centerCoeff );
        const auto lastElem = equations_[eqN+1].firstElem;
        for ( int ei = eq.firstElem; ei < lastElem; ++ ei )
        {
            const auto el = nonZeroElements_[ei];
            if ( freeVerts_.test( el.neiVert ) )
                mTriplets.emplace_back( n, freeVert2id_[el.neiVert], el.coeff );
        }
        ++n;
    }

    // equations for fixed neighbors of free vertices
    for ( auto v : firstLayerFixedVerts_ )
    {
        int eqN = regionVert2id_[v];
        const auto eq = equations_[eqN];
        const auto lastElem = equations_[eqN+1].firstElem;
        for ( int ei = eq.firstElem; ei < lastElem; ++ ei )
        {
            const auto el = nonZeroElements_[ei];
            if ( freeVerts_.test( el.neiVert ) )
                mTriplets.emplace_back( n, freeVert2id_[el.neiVert], el.coeff );
        }
        ++n;
    }
    assert( n == rowSz );

    M_.resize( rowSz, sz );
    M_.setFromTriplets( mTriplets.begin(), mTriplets.end() );

    SparseMatrix A = M_.adjoint() * M_;

    solver_->compute( A );
}

void Laplacian::updateRhs_()
{
    assert( solverValid_ );
    if ( rhsValid_ )
        return;
    rhsValid_ = true;

    MR_TIMER

    const auto rowSz = M_.rows();

    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( rowSz );

    // equations for free vertices
    int n = 0;
    for ( auto v : freeVerts_ )
    {
        assert( n == freeVert2id_[v] );
        int eqN = regionVert2id_[v];
        const auto eq = equations_[eqN];
        auto r = eq.rhs;
        const auto lastElem = equations_[eqN+1].firstElem;
        for ( int ei = eq.firstElem; ei < lastElem; ++ ei )
        {
            const auto el = nonZeroElements_[ei];
            if ( !freeVerts_.test( el.neiVert ) )
                r -= el.coeff * Vector3d{ mesh_.points[el.neiVert] };
        }
        for ( int i = 0; i < 3; ++i )
            rhs[i][n] = r[i];
        ++n;
    }

    // equations for fixed neighbors of free vertices
    for ( auto v : firstLayerFixedVerts_ )
    {
        int eqN = regionVert2id_[v];
        const auto eq = equations_[eqN];
        auto r = eq.rhs;
        r -= eq.centerCoeff * Vector3d{ mesh_.points[v] };
        const auto lastElem = equations_[eqN+1].firstElem;
        for ( int ei = eq.firstElem; ei < lastElem; ++ ei )
        {
            const auto el = nonZeroElements_[ei];
            if ( !freeVerts_.test( el.neiVert ) )
                r -= el.coeff * Vector3d{ mesh_.points[el.neiVert] };
        }
        for ( int i = 0; i < 3; ++i )
            rhs[i][n] = r[i];
        ++n;
    }
    assert( n == rowSz );

    tbb::parallel_for( tbb::blocked_range<int>( 0, 3, 1 ), [&]( const tbb::blocked_range<int> & range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
            rhs_[i] = M_.adjoint() * rhs[i];
    } );
}

void Laplacian::apply()
{
    MR_TIMER;
    if ( !freeVerts_.any() )
        return;
    updateSolver();

    Eigen::VectorXd sol[3];
    tbb::parallel_for( tbb::blocked_range<int>( 0, 3, 1 ), [&]( const tbb::blocked_range<int> & range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
            sol[i] = solver_->solve( rhs_[i] );
    } );

    // copy solution back into mesh points
    for ( auto v : freeVerts_ )
    {
        int mapv = freeVert2id_[v];
        auto & pt = mesh_.points[v];
        pt.x = (float) sol[0][mapv];
        pt.y = (float) sol[1][mapv];
        pt.z = (float) sol[2][mapv];
    }
}

TEST(MRMesh, Laplacian) 
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );

    {
        VertBitSet vs;
        vs.autoResizeSet( 0_v );
        Laplacian laplacian( sphere );
        laplacian.init( vs, Laplacian::EdgeWeights::Cotan );
        laplacian.apply();

        // fix the only free vertex
        laplacian.fixVertex( 0_v );
        laplacian.apply();
    }

    {
        Laplacian laplacian( sphere );
        // no free verts
        laplacian.init( {}, Laplacian::EdgeWeights::Cotan );
        laplacian.apply();
    }
}

} //namespace MR

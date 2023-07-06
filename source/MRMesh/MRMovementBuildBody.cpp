#include "MRMovementBuildBody.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRAffineXf3.h"
#include "MRBox.h"
#include "MRTimer.h"
#include "MRMatrix3.h"

namespace MR
{

Mesh makeMovementBuildBody( const Polyline3& body, const Polyline3& trajectory,
    const MovementBuildBodyParams& params )
{
    MR_TIMER;

    auto bodyContours = body.contours();
    auto trajectoryContours = trajectory.contours();
    // filter same points in trajectory
    for ( auto& trajC : trajectoryContours )
        trajC.erase( std::unique( trajC.begin(), trajC.end() ), trajC.end() );

    AffineXf3f xf;
    Vector3f trans;
    Matrix3f prevHalfRot;
    Matrix3f accumRot;
    auto rotationCenter = params.rotationCenter.value_or( body.computeBoundingBox().center() );
    Vector3f normal;
    if ( params.bodyNormal )
        normal = *params.bodyNormal;
    else
    {
        for ( const auto& bc : bodyContours )
        {
            if ( bc.size() < 3 )
                continue;
            for ( int i = 0; i + 1 < bc.size(); ++i )
                normal += cross( bc[i], bc[i + 1] );
            if ( bc.front() != bc.back() )
                normal += cross( bc.back(), bc.front() );
        }
    }
    // minus to have correct orientation of result mesh
    normal = -normal.normalized();

    Vector3f prevVec, nextVec; // needed for soft rotation

    auto halfRot = [] ( const Vector3f& from, const Vector3f& to )->Matrix3f
    {
        auto axis = cross( from, to );
        if ( axis.lengthSq() > 0 )
            return Matrix3f::rotation( axis, angle( from, to ) * 0.5f );
        if ( dot( from, to ) >= 0 )
            return {}; // identity matrix
        return Matrix3f::rotation( cross( from, from.furthestBasisVector() ), PI2_F );
    };

    Mesh res;
    int numEdges = 0;
    auto connectBlocks = [&] ( EdgeId newFirstEdge, EdgeId prevFirstEdge )
    {
        assert( numEdges != 0 );
        auto& tp = res.topology;
        EdgeId firstNewEdgeInLoop;
        for ( int i = 0; i < numEdges; ++i )
        {
            // first edge
            auto newEdge = tp.makeEdge();
            tp.splice( tp.prev( prevFirstEdge + i * 2 ), newEdge );
            tp.splice( newFirstEdge + i * 2, newEdge.sym() );

            if ( !firstNewEdgeInLoop )
                firstNewEdgeInLoop = newEdge;
            else
                tp.setLeft( newEdge.sym(), tp.addFaceId() );

            // diagonal edge
            newEdge = tp.makeEdge();
            tp.splice( tp.prev( prevFirstEdge + i * 2 ), newEdge );
            tp.splice( tp.prev( ( newFirstEdge + i * 2 ).sym() ), newEdge.sym() );
            tp.setLeft( newEdge.sym(), tp.addFaceId() );

            auto diagEdgePrev = tp.prev( newEdge.sym() );
            
            if ( diagEdgePrev == ( newFirstEdge + ( i + 1 ) * 2 ) )
                continue;
            if ( diagEdgePrev == firstNewEdgeInLoop.sym() )
            {
                tp.setLeft( firstNewEdgeInLoop.sym(), tp.addFaceId() );
                firstNewEdgeInLoop = {};
                continue;
            }
            // same as first if path is not finished
            // last in non closed path
            firstNewEdgeInLoop = {};
            newEdge = tp.makeEdge();
            tp.splice( ( prevFirstEdge + i * 2 ).sym(), newEdge );
            tp.splice( diagEdgePrev, newEdge.sym() );
            tp.setLeft( newEdge.sym(), tp.addFaceId() );
        }
    };
    auto getTrajectoryPoint = [&] ( const auto& cont, size_t i )
    {
        return params.t2bXf ? ( *params.t2bXf )( cont[i] ) : cont[i];
    };
    Vector3f basePos = getTrajectoryPoint( trajectoryContours[0], 0 );
    for ( const auto& trajectoryCont : trajectoryContours )
    {
        bool closed = trajectoryCont.size() > 2 && trajectoryCont.front() == trajectoryCont.back();
        EdgeId firstBodyEdge;
        EdgeId prevBodyEdge;
        bool initRotationDone = false;
        prevHalfRot = accumRot = Matrix3f();
        for ( int i = 0; i + ( closed ? 1 : 0 ) < trajectoryCont.size(); ++i )
        {
            const auto& trajPoint = getTrajectoryPoint( trajectoryCont, i );
            nextVec = prevVec = Vector3f();
            trans = trajPoint - basePos;
            if ( params.allowRotation )
            {
                if ( i > 0 )
                    prevVec = trajPoint - getTrajectoryPoint( trajectoryCont, i -1);
                else if ( closed )
                    prevVec = trajPoint - getTrajectoryPoint( trajectoryCont, trajectoryCont.size() - 2 );
                if ( i + 1 < trajectoryCont.size() )
                    nextVec = getTrajectoryPoint( trajectoryCont, i + 1 ) - trajPoint;
                else if ( closed )
                    prevVec = trajPoint - getTrajectoryPoint( trajectoryCont, 1 );

                if ( prevVec == Vector3f() )
                    prevVec = nextVec;
                if ( nextVec == Vector3f() )
                    nextVec = prevVec;
                
                if ( !initRotationDone )
                {
                    initRotationDone = true;
                    accumRot = Matrix3f::rotation( normal, prevVec );
                }

                auto curHalfRot = halfRot( prevVec, nextVec );
                accumRot = curHalfRot * prevHalfRot * accumRot;
                prevHalfRot = curHalfRot;
            }
            xf = AffineXf3f::translation( trans ) * AffineXf3f::xfAround( accumRot, rotationCenter );

            auto curBodyEdge = res.addSeparateContours( bodyContours, &xf );
            if ( !firstBodyEdge )
                firstBodyEdge = curBodyEdge;

            if ( prevBodyEdge )
            {
                if ( numEdges == 0 )
                    numEdges = ( curBodyEdge - prevBodyEdge ) / 2;
                connectBlocks( curBodyEdge, prevBodyEdge );
            }
            prevBodyEdge = curBodyEdge;
        }
        if ( closed )
            connectBlocks( firstBodyEdge, prevBodyEdge );
    }

    return res;
}

}
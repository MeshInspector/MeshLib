#pragma once

#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshProject.h"

namespace MR
{

template<typename T>
T calcNewVertAttribute( const Mesh& newMesh, const Mesh& oldMesh, const T& data, ProgressCallback progressCb )
{
    T newData;
    if ( !data.empty() )
    {
        newData.resize( newMesh.topology.lastValidVert() + 1 );
        BitSetParallelFor( newMesh.topology.getValidVerts(), [&] ( VertId id )
            {
                auto res = findProjection( newMesh.points[id], oldMesh ).mtp;
                VertId v1 = oldMesh.topology.org( res.e );
                VertId v2 = oldMesh.topology.dest( res.e );
                VertId v3 = oldMesh.topology.dest( oldMesh.topology.next( res.e ) );
                newData[id] = res.bary.interpolate( data[v1], data[v2], data[v3] );
            },
        progressCb );
    }

    return newData;
}

//template<typename T>
//T calcNewFaceAttribute( const Mesh& newMesh, const Mesh& oldMesh, const T& data, ProgressCallback progressCb )
//{
//    T newData;
//    if ( !data.empty() )
//    {
//        newData.resize( newMesh->topology.lastValidFace() + 1 );
//        BitSetParallelFor( newMesh->topology.lastValidFace(), [&] ( FaceId id )
//        {
//            auto res = findProjection( newMesh->points[id], *oldMesh ).mtp;
//            VertId v1 = oldMesh->topology.org( res.e );
//            VertId v2 = oldMesh->topology.dest( res.e );
//            VertId v3 = oldMesh->topology.dest( oldMesh->topology.next( res.e ) );
//            newData[id] = res.bary.interpolate( data[v1], data[v2], data[v3] );
//        },
//        progressCb );
//    }
//
//    return newData;
//}

}

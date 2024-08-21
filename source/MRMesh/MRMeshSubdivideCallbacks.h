#pragma once

#include "MRMesh/MRMesh.h"
#include "MRMeshDecimateCallbacks.h"

namespace MR
{

template <typename T>
auto onEdgeSplitVertAttribute( const Mesh& mesh, Vector<T, VertId>& data );

template <typename T>
auto onEdgeSplitFaceAttribute( const Mesh& mesh, Vector<T, VertId>& data );

MRMESH_API OnEdgeSplit meshOnEdgeSplitAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params );

MRMESH_API OnEdgeSplit meshOnEdgeSplitVertAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params );

MRMESH_API OnEdgeSplit meshOnEdgeSplitFaceAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params );

template <typename T>
auto onEdgeSplitVertAttribute( const Mesh& mesh, Vector<T, VertId>& data )
{
    auto preCollapse = [&] ( EdgeId e1, EdgeId e )
    {
        const auto org = mesh.topology.org( e1 );
        const auto dest = mesh.topology.dest( e );
        if ( org < data.size() && dest < data.size() )
            data.push_back( ( data[org] + data[dest] ) * 0.5f );
    };

    return preCollapse;
}

template <typename T>
auto onEdgeSplitFaceAttribute( const Mesh& mesh, Vector<T, FaceId>& data )
{
    auto preCollapse = [&] ( EdgeId e1, EdgeId e )
    {
        auto oldLeft = mesh.topology.left( e.undirected() );
        auto newLeft = mesh.topology.left( e1.undirected() );

        FaceId existing;
        if ( oldLeft < data.size() )
            existing = oldLeft;
        else if ( newLeft < data.size() )
            existing = newLeft;
        else
            existing = FaceId( 0 );

        data.autoResizeSet( newLeft, data[existing] );

        //if ( oldLeft < data.size() )
        //if ( oldRight < data.size() )

        auto oldRight = mesh.topology.right( e.undirected() );
        auto newRight = mesh.topology.right( e1.undirected() );

        if ( oldRight < data.size() )
            existing = oldRight;
        else if ( newRight < data.size() )
            existing = newRight;
        else
            existing = FaceId( 0 );

        data.autoResizeSet( newRight, data[existing] );
    };

    return preCollapse;
}

}

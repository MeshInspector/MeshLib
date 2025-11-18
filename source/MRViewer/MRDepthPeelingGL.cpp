#include "MRDepthPeelingGL.h"
#include "MRViewer.h"
#include "MRSceneTextureGL.h"
#include "MRViewport.h"
#include "MRMesh\MRSceneRoot.h"

namespace MR
{

void DepthPeelingGL::reset( const Vector2i& size )
{
    if ( accumFB_.getColorTexture() != GlTexture2::NO_TEX )
    {
        accumFB_.del();
        qt_.del();
    }
    accumFB_.gen( size, true, 0, true );
    qt_.gen();
}

void DepthPeelingGL::draw()
{
    accumFB_.draw( qt_, { .size = accumFB_.getSize(),.wrap = WrapType::Clamp,.filter = FilterType::Discrete } );
}

bool DepthPeelingGL::doPasses( SceneTextureGL* sceneTexture )
{
    assert( sceneTexture );
    if ( !sceneTexture )
        return false;
    assert( sceneTexture->isBound() );
    
    sceneTexture->copyTexture();

    int numTransparent = 0;
    GL_EXEC( glClearDepth( 0.0f ) );
    accumFB_.bind( true ); // reset depth buffer to 0.0 value for first pass
    GL_EXEC( glClearDepth( 1.0f ) );

    for ( int i = 0; i < numPasses_; ++i )
    {
        accumFB_.copyTextureBindDef();
        accumFB_.bind( false );
        GL_EXEC( glClear( GL_DEPTH_BUFFER_BIT ) );
        for ( const auto& viewport : getViewerInstance().viewport_list )
        {
            viewport.recursiveDraw( SceneRoot::get(), DepthFunction::Default, AffineXf3f(), RenderModelPassMask::Transparent, 
                { sceneTexture->getDepthTextureId(),accumFB_.getColorTexture(), accumFB_.getDepthTexture() }, 
                &numTransparent );
        }
        if ( numTransparent == 0 )
            break;
    }
    if ( numTransparent != 0 )
        accumFB_.copyTextureBindDef();
    sceneTexture->bind( false );

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    return numTransparent != 0;
}

}
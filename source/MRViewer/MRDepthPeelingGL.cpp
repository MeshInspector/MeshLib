#include "MRDepthPeelingGL.h"
#include "MRViewer.h"
#include "MRSceneTextureGL.h"
#include "MRViewport.h"
#include "MRMesh\MRSceneRoot.h"

namespace MR
{

void DepthPeelingGL::reset( const Vector2i& size, int msaa )
{
    if ( accumFB_[0].getColorTexture() != GlTexture2::NO_TEX )
    {
        accumFB_[0].del();
        //accumFB_[1].del();
        qt_.del();
    }
    accumFB_[0].gen( size, true, msaa );
    //accumFB_[1].gen( size, false, msaa );
    qt_.gen();
}

void DepthPeelingGL::draw()
{
    accumFB_[0].draw( qt_, { .size = accumFB_[0].getSize(),.wrap = WrapType::Clamp,.filter = FilterType::Discrete } );
}

bool DepthPeelingGL::doPasses( SceneTextureGL* sceneTexture )
{
    assert( sceneTexture );
    if ( !sceneTexture )
        return false;
    assert( sceneTexture->isBound() );
    
    sceneTexture->copyTexture();

    int numTransparent = 0;
    accumFB_[0].bind( true );
    for ( int i = 0; i < 1; ++i )
    {
        accumFB_[0].copyTextureBindDef();
        accumFB_[0].bind( false );
        for ( const auto& viewport : getViewerInstance().viewport_list )
        {
            viewport.recursiveDraw( SceneRoot::get(), DepthFunction::Default, AffineXf3f(), RenderModelPassMask::Transparent, 
                { sceneTexture->getDepthTextureId(),accumFB_[0].getColorTexture(), accumFB_[0].getDepthTexture() }, 
                &numTransparent );
        }
        if ( numTransparent == 0 )
            break;
    }
    if ( numTransparent != 0 )
        accumFB_[0].copyTextureBindDef();
    sceneTexture->bind( false );

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    return numTransparent != 0;
}

}
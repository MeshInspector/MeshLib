#include "MRDepthPeelingGL.h"
#include "MRViewer.h"
#include "MRSceneTextureGL.h"
#include "MRViewport.h"
#include "MRMesh/MRSceneRoot.h"

namespace MR
{

void DepthPeelingGL::reset( const Vector2i& size )
{
    if ( accumFB_.getColorTexture() != GlTexture2::NO_TEX )
    {
        accumFB_.del();
        qt_.del();
        nearestDepth_.del();
        GL_EXEC( glDeleteFramebuffers( 1, &nearestDepthFB_ ) );
    }
    accumFB_.gen( size, true, 0, true );
    qt_.gen();

    nearestDepth_.gen();
    nearestDepth_.loadData( { .resolution = GlTexture2::ToResolution( size ), .internalFormat = GL_DEPTH_COMPONENT32F,
        .format = GL_DEPTH_COMPONENT, .type = GL_FLOAT, .wrap = WrapType::Clamp, .filter = FilterType::Discrete }, ( const char* )nullptr );
    GL_EXEC( glGenFramebuffers( 1, &nearestDepthFB_ ) );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, nearestDepthFB_ ) );
    GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, nearestDepth_.getId(), 0 ) );
    const GLenum noColor = GL_NONE;
    GL_EXEC( glDrawBuffers( 1, &noColor ) );
    GL_EXEC( glReadBuffer( GL_NONE ) );
    assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
}

void DepthPeelingGL::draw()
{
    // only on the samples where the nearest layer is not behind the opaque scene,
    // so the antialiased edges of closer opaque objects stay intact
    GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    GL_EXEC( glDepthFunc( GL_LEQUAL ) );
    GL_EXEC( glDepthMask( GL_FALSE ) );
    accumFB_.draw( qt_, { .size = accumFB_.getSize(),.wrap = WrapType::Clamp,.filter = FilterType::Discrete,
        .depthTexture = numPasses_ > 1 ? nearestDepth_.getId() : 0, .maxDepth = 1.0f } );
    GL_EXEC( glDepthMask( GL_TRUE ) );
    GL_EXEC( glDepthFunc( GL_LESS ) );
}

bool DepthPeelingGL::doPasses( FramebufferData* bgFramebuffer )
{
    assert( bgFramebuffer );
    if ( !bgFramebuffer )
        return false;
    assert( bgFramebuffer->isBound() );
    
    bgFramebuffer->copyTextureBindDef();

    int numTransparent = 0;

    accumFB_.bind( true, 0.0f ); // reset depth buffer to 0.0 value for first pass

    for ( int i = 0; i < numPasses_; ++i )
    {
        accumFB_.copyTextureBindDef();
        if ( i == 1 )
        {
            // keep the depth of the nearest layer for `draw`
            const auto& size = accumFB_.getSize();
            GL_EXEC( glBindFramebuffer( GL_READ_FRAMEBUFFER, accumFB_.getCopyFramebuffer() ) );
            GL_EXEC( glBindFramebuffer( GL_DRAW_FRAMEBUFFER, nearestDepthFB_ ) );
            GL_EXEC( glBlitFramebuffer( 0, 0, size.x, size.y, 0, 0, size.x, size.y, GL_DEPTH_BUFFER_BIT, GL_NEAREST ) );
        }
        accumFB_.bind( false );
        float cOneValue = 1.0f;
        GL_EXEC( glClearBufferfv( GL_DEPTH, 0, &cOneValue ) );
        for ( const auto& viewport : getViewerInstance().viewport_list )
        {
            viewport.recursiveDraw( SceneRoot::get(), DepthFunction::Default, AffineXf3f(), RenderModelPassMask::Transparent, 
                { bgFramebuffer->getDepthTexture(),accumFB_.getColorTexture(), accumFB_.getDepthTexture() },
                &numTransparent );
        }
        if ( numTransparent == 0 )
            break;
    }
    if ( numTransparent != 0 )
        accumFB_.copyTextureBindDef();
    bgFramebuffer->bind( false );

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    return numTransparent != 0;
}

}
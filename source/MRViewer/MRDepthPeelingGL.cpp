#include "MRDepthPeelingGL.h"
#include "MRViewer.h"
#include "MRSceneTextureGL.h"
#include "MRViewport.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRMesh/MRSceneRoot.h"

namespace MR
{

void DepthPeelingGL::reset( const Vector2i& size )
{
    if ( accumFB_.getColorTexture() != GlTexture2::NO_TEX )
    {
        accumFB_.del();
        qt_.del();
        for ( auto& t : targets_ )
        {
            GL_EXEC( glDeleteFramebuffers( 1, &t.fbo ) );
            t.color.del();
            t.depth.del();
        }
    }
    accumFB_.gen( size, true, 0, true );
    qt_.gen();
    const auto resolution = GlTexture2::ToResolution( size );
    for ( auto& t : targets_ )
    {
        GL_EXEC( glGenFramebuffers( 1, &t.fbo ) );
        GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, t.fbo ) );
        t.color.gen();
        t.color.loadData( { .resolution = resolution, .wrap = WrapType::Clamp, .filter = FilterType::Discrete }, ( const char* )nullptr );
        GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.color.getId(), 0 ) );
        t.depth.gen();
        t.depth.loadData( { .resolution = resolution, .internalFormat = GL_DEPTH_COMPONENT32F, .format = GL_DEPTH_COMPONENT,
            .type = GL_FLOAT, .wrap = WrapType::Clamp, .filter = FilterType::Discrete }, ( const char* )nullptr );
        GL_EXEC( glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, t.depth.getId(), 0 ) );
        assert( glCheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE );
    }
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
}

void DepthPeelingGL::draw()
{
    // back to front, each layer only on the samples where it is not behind the opaque scene (as in the peeling),
    // so the antialiased edges of closer opaque objects stay intact
    GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    GL_EXEC( glDepthFunc( GL_LEQUAL ) );
    GL_EXEC( glDepthMask( GL_FALSE ) );
    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );
    for ( int i = std::min( numPasses_, cSeparateLayers + 1 ) - 1; i >= 0; --i )
    {
        const bool accumulated = i == cSeparateLayers && numPasses_ > cSeparateLayers + 1;
        drawQuad_( accumulated ? accumFB_.getColorTexture() : targets_[i].color.getId(), targets_[i].depth.getId(), false );
    }
    GL_EXEC( glDepthMask( GL_TRUE ) );
}

bool DepthPeelingGL::doPasses( FramebufferData* bgFramebuffer )
{
    assert( bgFramebuffer );
    if ( !bgFramebuffer )
        return false;
    assert( bgFramebuffer->isBound() );
    
    bgFramebuffer->copyTextureBindDef();

    // the resolved depth keeps one sample of an antialiased edge, so peel against the farthest depth around the pixel:
    // a layer hidden on some samples of the pixel is still needed on the others, and `draw` hides it per sample
    const auto& bgDepth = targets_[cSeparateLayers + 1];
    GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, bgDepth.fbo ) );
    GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    GL_EXEC( glDepthFunc( GL_ALWAYS ) );
    GL_EXEC( glDepthMask( GL_TRUE ) );
    drawQuad_( bgFramebuffer->getColorTexture(), bgFramebuffer->getDepthTexture(), true );
    GL_EXEC( glDepthFunc( GL_LESS ) );

    int numTransparent = 0;

    accumFB_.bind( true, 0.0f ); // empty color and zero depth: the input of the first pass
    accumFB_.copyTextureBindDef();

    const float cOneValue = 1.0f;
    for ( int i = 0; i < numPasses_; ++i )
    {
        auto fgColor = accumFB_.getColorTexture();
        auto fgDepth = i == 0 ? accumFB_.getDepthTexture() : targets_[std::min( i - 1, cSeparateLayers )].depth.getId();
        if ( i <= cSeparateLayers )
        {
            // the nearest layers and the first of the farther ones: each alone in its target
            GL_EXEC( glBindFramebuffer( GL_FRAMEBUFFER, targets_[i].fbo ) );
            const float cZeroColor[4] = {};
            GL_EXEC( glClearBufferfv( GL_COLOR, 0, cZeroColor ) );
        }
        else
        {
            // the farther layers are accumulated behind the first of them
            if ( i == cSeparateLayers + 1 )
                fgColor = targets_[cSeparateLayers].color.getId();
            else
            {
                accumFB_.copyTextureBindDef();
                fgDepth = accumFB_.getDepthTexture();
            }
            accumFB_.bind( false );
        }
        GL_EXEC( glClearBufferfv( GL_DEPTH, 0, &cOneValue ) );
        for ( const auto& viewport : getViewerInstance().viewport_list )
        {
            viewport.recursiveDraw( SceneRoot::get(), DepthFunction::Default, AffineXf3f(), RenderModelPassMask::Transparent, 
                { bgDepth.depth.getId(), fgColor, fgDepth },
                &numTransparent );
        }
        if ( numTransparent == 0 )
            break;
    }
    if ( numTransparent != 0 && numPasses_ > cSeparateLayers + 1 )
        accumFB_.copyTextureBindDef();
    bgFramebuffer->bind( false );

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    return numTransparent != 0;
}

void DepthPeelingGL::drawQuad_( unsigned colorTex, unsigned depthTex, bool dilate )
{
    const auto& size = accumFB_.getSize();
    GL_EXEC( glViewport( 0, 0, size.x, size.y ) );
    const auto shader = GLStaticHolder::getShaderId( GLStaticHolder::DepthPeelQuad );
    GL_EXEC( glUseProgram( shader ) );
    qt_.bind();
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, colorTex ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "colors" ), 0 ) );
    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    GL_EXEC( glBindTexture( GL_TEXTURE_2D, depthTex ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "depths" ), 1 ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "dilate" ), dilate ) );
    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 6 ) );
}

}

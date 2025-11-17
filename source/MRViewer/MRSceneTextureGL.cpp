#include "MRSceneTextureGL.h"
#include "MRGladGlfw.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRViewer.h"

namespace MR
{

void SceneTextureGL::bind( bool clear )
{
    fd_.bind( clear );
}

void SceneTextureGL::unbind()
{
    fd_.bindDefault();
}

void SceneTextureGL::reset( const Vector2i& size, int msaa )
{
    if ( fd_.getColorTexture() != GlTexture2::NO_TEX )
    {
        fd_.del();
        qt_.del();
    }
    fd_.gen( size, false, msaa );
    qt_.gen();
}

void SceneTextureGL::copyTexture()
{
    fd_.copyTextureBindDef();
}

void SceneTextureGL::draw()
{
    fd_.draw( qt_, { .size = fd_.getSize(),.wrap = WrapType::Mirror,.filter = FilterType::Discrete } );
}

}
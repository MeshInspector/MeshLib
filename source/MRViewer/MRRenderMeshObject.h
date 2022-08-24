#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRBuffer.h"
#include "MRRenderGLHelpers.h"

namespace MR
{
class RenderMeshObject : public IRenderObject
{
public:
    RenderMeshObject( const VisualObject& visObj );
    ~RenderMeshObject();

    virtual void render( const RenderParams& params ) const override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const override;
    virtual size_t heapBytes() const override;

private:
    const ObjectMeshHolder* objMesh_;

    // need this to use per corner rendering (this is not simple copy of mesh vertices etc.)
    mutable Buffer<Vector3f> vertPosBufferObj_;
    mutable Buffer<Vector3f> vertNormalsBufferObj_;
    mutable Buffer<Color> vertColorsBufferObj_;
    mutable Buffer<UVCoord> vertUVBufferObj_;
    mutable Buffer<Vector3i> facesIndicesBufferObj_;
    mutable Buffer<Vector2i> edgesIndicesBufferObj_;
    mutable Buffer<unsigned> faceSelectionTexture_;
    mutable Buffer<Vector4f> faceNormalsTexture_;
    mutable std::vector<Vector3f> borderHighlightPoints_;
    mutable std::vector<Vector3f> selectedEdgesPoints_;

    typedef unsigned int GLuint;

    GLuint borderArrayObjId_{ 0 };
    GLuint borderBufferObjId_{ 0 };

    GLuint selectedEdgesArrayObjId_{ 0 };
    GLuint selectedEdgesBufferObjId_{ 0 };

    GLuint meshArrayObjId_{ 0 };
    GLuint meshPickerArrayObjId_{ 0 };

    mutable GlBuffer vertPosBuffer_;
    mutable GlBuffer vertUVBuffer_;
    mutable GlBuffer vertNormalsBuffer_;
    mutable GlBuffer vertColorsBuffer_;

    GLuint facesIndicesBufferObjId_{ 0 };
    GLuint edgesIndicesBufferObjId_{ 0 };
    GLuint texture_{ 0 };

    GLuint faceSelectionTex_{ 0 };

    GLuint faceColorsTex_{ 0 };

    GLuint facesNormalsTex_{ 0 };

    int maxTexSize_{ 0 };

    void renderEdges_( const RenderParams& parameters, GLuint vao, GLuint vbo, const std::vector<Vector3f>& data,
        GLuint count, const Color& color, unsigned dirtyValue ) const;

    void renderMeshEdges_( const RenderParams& parameters ) const;

    void bindMesh_( bool alphaSort ) const;
    void bindMeshPicker_() const;

    void drawMesh_( bool solid, ViewportId viewportId, bool picker = false ) const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_( ViewportId id ) const;
    void updateMeshEdgesBuffer_() const;
    void updateBorderLinesBuffer_() const;
    void updateSelectedEdgesBuffer_() const;

    void resetBuffers_() const;

    // Marks dirty buffers that need to be uploaded to OpenGL
    mutable uint32_t dirty_;
    mutable bool meshFacesDirty_{ false };
    mutable bool meshEdgesDirty_{ false };
    // this is needed to fix case of missing normals bind (can happen if `renderPicker` before first `render` with flat shading)
    mutable bool normalsBound_{ false };
    // store element counts separately because the buffers could be cleared
    mutable size_t vertsCount_{ 0 };
    mutable size_t vertNormalsCount_{ 0 };
    mutable size_t vertColorsCount_{ 0 };
    mutable size_t vertUVCount_{ 0 };
    mutable int meshFacesCount_{ 0 };
    mutable int meshEdgesCount_{ 0 };
    mutable int borderPointsCount_{ 0 };
    mutable int selectedPointsCount_{ 0 };
};

}
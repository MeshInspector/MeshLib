#pragma once
#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRMeshNormals.h"

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

    virtual const Vector<Vector3f, FaceId>& getFacesNormals() const;
    virtual const Vector<TriangleCornerNormals, FaceId>& getCornerNormals() const;

private:
    const ObjectMeshHolder* objMesh_;
    mutable std::mutex readCacheMutex_;
    // need this to use per corner rendering (this is not simple copy of mesh vertices etc.)
    mutable std::vector<Vector3f> vertPosBufferObj_;
    mutable std::vector<Vector3f> vertNormalsBufferObj_;
    mutable std::vector<Color> vertColorsBufferObj_;
    mutable std::vector<UVCoord> vertUVBufferObj_;
    mutable std::vector<Vector3i> facesIndicesBufferObj_;
    mutable std::vector<Vector2i> edgesIndicesBufferObj_;
    mutable std::vector<unsigned> faceSelectionTexture_;
    mutable std::vector<Vector4f> faceNormalsTexture_;
    mutable std::vector<Vector3f> borderHighlightPoints_;
    mutable std::vector<Vector3f> selectedEdgesPoints_;
    mutable Vector<TriangleCornerNormals, FaceId> cornerNormalsCache_;
    mutable Vector<Vector3f, FaceId> facesNormalsCache_;

    typedef unsigned int GLuint;

    GLuint borderArrayObjId_{ 0 };
    GLuint borderBufferObjId_{ 0 };

    GLuint selectedEdgesArrayObjId_{ 0 };
    GLuint selectedEdgesBufferObjId_{ 0 };

    GLuint meshArrayObjId_{ 0 };
    GLuint meshPickerArrayObjId_{ 0 };

    GLuint vertPosBufferObjId_{ 0 };
    GLuint vertUVBufferObjId_{ 0 };
    GLuint vertNormalsBufferObjId_{ 0 };
    GLuint vertColorsBufferObjId_{ 0 };

    GLuint facesIndicesBufferObjId_{ 0 };
    GLuint edgesIndicesBufferObjId_{ 0 };
    GLuint texture_{ 0 };

    GLuint faceSelectionTex_{ 0 };

    GLuint faceColorsTex_{ 0 };

    GLuint facesNormalsTex_{ 0 };

    void renderEdges_( const RenderParams& parameters, GLuint vao, GLuint vbo, const std::vector<Vector3f>& data,
        const Color& color, unsigned dirtyValue ) const;

    void renderMeshEdges_( const RenderParams& parameters ) const;

    void bindMesh_( bool alphaSort ) const;
    void bindMeshPicker_() const;

    void drawMesh_( bool solid, ViewportId viewportId, bool picker = false ) const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_( ViewportId id ) const;

    virtual Vector<Vector3f, FaceId> computeFacesNormals_() const;
    virtual Vector<TriangleCornerNormals, FaceId> computeCornerNormals_() const;

    // Marks dirty buffers that need to be uploaded to OpenGL
    mutable uint32_t dirty_;
    mutable bool meshEdgesDirty_{ false };
    // this is needed to fix case of missing normals bind (can happen if `renderPicker` before first `render` with flat shading)
    mutable bool normalsBound_{ false };
};

}
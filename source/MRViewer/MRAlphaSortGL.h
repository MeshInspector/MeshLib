#pragma once

namespace MR
{

class AlphaSortGL
{
public:
    typedef unsigned int GLuint;

    ~AlphaSortGL();

    // Initialize all GL buffers and arrays
    void init();
    // Free all GL data
    void free();

    // Set all textures used for alpha sorting to zero (heads texture, atomics texture)
    void clearTransparencyTextures() const;
    // Draws alpha sorting overlay quad texture to screen
    void drawTransparencyTextureToScreen() const;
    // Updates size of textures used in alpha sorting (according to viewport size)
    void updateTransparencyTexturesSize( int width, int height );

private:
    bool inited_ = false;
    int width_{ 0 };
    int height_{ 0 };

    GLuint transparency_quad_vbo = 0;
    GLuint transparency_quad_vao = 0;
    GLuint transparency_heads_texture_vbo = 0;
    GLuint transparency_shared_shader_data_vbo = 0;
    GLuint transparency_atomic_counter_vbo = 0;
    GLuint transparency_static_clean_vbo = 0;
};

}
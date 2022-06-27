#pragma once

namespace MR
{
template<typename T>
GLint bindVertexAttribArray(
    const GLuint program_shader,
    const std::string& name,
    GLuint bufferID,
    const std::vector<T>& V,
    int baseTypeElementsNumber,
    bool refresh )
{
    GL_EXEC( GLint id = glGetAttribLocation( program_shader, name.c_str() ) );
    if ( id < 0 )
        return id;
    if ( V.size() == 0 )
    {
        GL_EXEC( glDisableVertexAttribArray( id ) );
        return id;
    }
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, bufferID ) );
    if ( refresh )
    {
        GL_EXEC( glBufferData( GL_ARRAY_BUFFER, sizeof( T ) * V.size(), V.data(), GL_DYNAMIC_DRAW ) );
    }

    // GL_FLOAT is left here consciously 
    if constexpr ( std::is_same_v<Color, T> )
    {
        GL_EXEC( glVertexAttribPointer( id, baseTypeElementsNumber, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0 ) );
    }
    else
    {
        GL_EXEC( glVertexAttribPointer( id, baseTypeElementsNumber, GL_FLOAT, GL_FALSE, 0, 0 ) );
    }

    GL_EXEC( glEnableVertexAttribArray( id ) );
    return id;
}
}
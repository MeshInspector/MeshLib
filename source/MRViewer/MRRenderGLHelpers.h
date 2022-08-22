#pragma once

namespace MR
{
template<typename T, template<typename, typename...> class C, typename... args>
GLint bindVertexAttribArray(
    const GLuint program_shader,
    const std::string& name,
    GLuint bufferID,
    const C<T, args...>& V,
    int baseTypeElementsNumber,
    bool refresh,
    bool forceUse = false )
{
    GL_EXEC( GLint id = glGetAttribLocation( program_shader, name.c_str() ) );
    if ( id < 0 )
        return id;
    if ( V.size() == 0 && !forceUse )
    {
        GL_EXEC( glDisableVertexAttribArray( id ) );
        return id;
    }
    GL_EXEC( glBindBuffer( GL_ARRAY_BUFFER, bufferID ) );
    if ( refresh )
    {
        GLint64 bufSize = sizeof( T ) * V.size();
        auto maxUploadSize = ( GLint64( 1 ) << 32 ) - 4096; //4Gb - 4096, 4Gb is already too much
        if ( bufSize <= maxUploadSize )
        {
            // buffers less than 4Gb are ok to load immediately
            GL_EXEC( glBufferData( GL_ARRAY_BUFFER, bufSize, V.data(), GL_DYNAMIC_DRAW ) );
        }
        else
        {
            // buffers more than 4Gb are better to split on chunks to avoid strange errors from GL or drivers
            GL_EXEC( glBufferData( GL_ARRAY_BUFFER, bufSize, nullptr, GL_DYNAMIC_DRAW ) );
            GLint64 remStart = 0;
            auto remSize = bufSize;
            for ( ; remSize > maxUploadSize; remSize -= maxUploadSize, remStart += maxUploadSize )
            {
                GL_EXEC( glBufferSubData( GL_ARRAY_BUFFER, remStart, maxUploadSize, (const char *)V.data() + remStart ) );
            }
            GL_EXEC( glBufferSubData( GL_ARRAY_BUFFER, remStart, remSize, (const char *)V.data() + remStart ) );
        }
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
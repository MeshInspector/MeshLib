#include "MRCreateShader.h"
#include "MRGLMacro.h"
#include "MRPch/MRSpdlog.h"
#include <sstream>
#include <string>
#include <GLFW/glfw3.h>

namespace
{
GLuint compileShader( const std::string& vertSource, const std::string& fragSource, std::string& out )
{
    out.clear();
    if ( vertSource == "" || fragSource == "" )
    {
        spdlog::warn( "Cannot create empty shader" );
        return 0;
    }

    // create program
    GL_EXEC( GLuint id = glCreateProgram() );
    if ( id == 0 )
    {
        spdlog::warn( "Cannot create shader" );
        return 0;
    }
    GLuint f = 0, v = 0;

    auto loadShader = [id]( const std::string& source, GLenum type, GLuint& s )->bool
    {
        GL_EXEC( s = glCreateShader( type ) );
        if ( s == 0 )
            return false;
        // Pass shader source string
        const char* c = source.c_str();
        GL_EXEC( glShaderSource( s, 1, &c, NULL ) );
        GL_EXEC( glCompileShader( s ) );
        GL_EXEC( glAttachShader( id, s ) );

		GLint infologLength = 0;
		GLint charsWritten = 0;
		char* infoLog;
		// Get shader info log from opengl
		GL_EXEC( glGetShaderiv( s, GL_INFO_LOG_LENGTH, &infologLength ) );
		// Only print if there is something in the log
		if (infologLength > 1)
		{
			infoLog = (char*)malloc(infologLength);
			GL_EXEC( glGetShaderInfoLog( s, infologLength, &charsWritten, infoLog ) );
			std::string compileLog = std::string( infoLog );
			free(infoLog);
			spdlog::critical( compileLog );
		}

        return true;
    };
    if ( !loadShader( vertSource, GL_VERTEX_SHADER, v ) )
    {
        spdlog::warn( "Cannot compile vertex shader" );
        return 0;
    }
    if ( !loadShader( fragSource, GL_FRAGMENT_SHADER, f ) )
    {
        spdlog::warn( "Cannot compile fragment shader" );
        return 0;
    }

    // Link program
    GL_EXEC( glLinkProgram( id ) );
    const auto& detach = [&id]( const GLuint shader )
    {
        if ( shader )
        {
            GL_EXEC( glDetachShader( id, shader ) );
            GL_EXEC( glDeleteShader( shader ) );
        }
    };
    detach( f );
    detach( v );

    // print log if any
    GLint infologLength = 0;
    GLint charsWritten = 0;
    char* infoLog;
    GL_EXEC( glGetProgramiv( id, GL_INFO_LOG_LENGTH, &infologLength ) );
    if ( infologLength > 1 )
    {
        infoLog = (char*) malloc( infologLength );
        GL_EXEC( glGetProgramInfoLog( id, infologLength, &charsWritten, infoLog ) );
        out = std::string( infoLog );
        free( infoLog );
    }

    return id;
}
}

namespace MR
{

void createShader( [[maybe_unused]]const std::string& shader_name,
    const std::string& vert_source,
    const std::string& frag_source,
    GLuint& prog_id,
    const DisabledWarnings& suppressedWarns )
{
#ifdef LOG_SHADERS
    spdlog::info( "Creating shader: {}", shader_name );
#endif
    std::string shaderLog;
    prog_id = compileShader( vert_source, frag_source, shaderLog );
    if ( shaderLog.empty() )
        return;
    std::stringstream ss( shaderLog );
    std::string line;
    while ( std::getline( ss, line, '\n' ) )
    {
        if ( line.empty() )
            continue;
        auto warnPos = line.find( "warning" );
        if ( warnPos != std::string::npos )
        {
            if ( suppressedWarns.empty() )
                spdlog::warn( line );
            else
            {
                int warnCode = std::atoi( line.substr( warnPos + 9, 4 ).c_str() );
                if ( std::find( suppressedWarns.begin(), suppressedWarns.end(), warnCode ) == suppressedWarns.end() )
                    spdlog::warn( line );
            }
        }
        else if ( line.find( "error" ) != std::string::npos )
            spdlog::error( line );
    }
}

void destroyShader( GLuint id )
{
    // Don't try to destroy id == 0 (no shader program)
    if ( id == 0 )
    {
        spdlog::warn( "Destroy shader: shader id should be non zero." );
        return ;
    }
    // Get each attached shader one by one and detach and delete it
    GLsizei count = 0;
    // shader id
    GLuint s;
    do
    {
        // Try to get at most *1* attached shader
        GL_EXEC( glGetAttachedShaders( id, 1, &count, &s ) );

        // Check that we actually got *1*
        if ( count == 1 )
        {
            // Detach and delete this shader
            GL_EXEC( glDetachShader( id, s ) );
            GL_EXEC( glDeleteShader( s ) );
        }
    } while ( count > 0 );
    // Now that all of the shaders are gone we can just delete the program
    GL_EXEC( glDeleteProgram( id ) );
}

}

#pragma once
#include "exports.h"
#include "MRGladGlfw.h"
#include <string>
#include <vector>


namespace MR
{

struct ShaderWarning
{
	// number is presented in some of gpu shader compilations output
	int number{0};
	// some part of warning line to find
	std::string line;
};

using DisabledWarnings = std::vector<ShaderWarning>;

// This function creates shader and logs output
MRVIEWER_API void createShader( const std::string& shader_name,
	const std::string& vert_source,
	const std::string& frag_source,
	GLuint& prog_id,
	const DisabledWarnings& suppressedWarns = {} );

// Destroys shader program
MRVIEWER_API void destroyShader( GLuint id );
}

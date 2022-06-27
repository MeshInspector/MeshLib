#pragma once
#include <string>
#include <vector>
#include "MRGladGlfw.h"


namespace MR
{

using DisabledWarnings = std::vector<int>;

// This function creates shader and logs output
void createShader( const std::string& shader_name,
	const std::string& vert_source,
	const std::string& frag_source,
	GLuint& prog_id,
	const DisabledWarnings& suppressedWarns = {} );

// Destroys shader program
void destroyShader( GLuint id );
}

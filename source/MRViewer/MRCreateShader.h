#pragma once
#include <string>
#include <vector>
#include "MRGladGlfw.h"


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
void createShader( const std::string& shader_name,
	const std::string& vert_source,
	const std::string& frag_source,
	GLuint& prog_id,
	const DisabledWarnings& suppressedWarns = {} );

// Destroys shader program
void destroyShader( GLuint id );
}

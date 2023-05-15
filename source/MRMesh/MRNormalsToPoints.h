#pragma once

#include "MRMeshFwd.h"
#include <memory>

namespace MR
{

class NormalsToPoints
{
public:
    /// builds linear system and prepares a solver for it
    MRMESH_API void prepare( const MeshTopology & topology );

    class ISolver
    {
    public:
        virtual ~ISolver() = default;
        virtual void prepare( const MeshTopology & topology ) = 0;
    };
private:
    std::unique_ptr<ISolver> solver_;
};

} //namespace MR

#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRChangeXfAction.h"

namespace MR
{

class MRVIEWER_CLASS DirectionWidget
{
    std::shared_ptr<ObjectMesh> directionObj_;
    Vector3f dir_;
    Vector3f base_;
    float length_;

    class ChangeDirAction : public ChangeXfAction
    {
    public:
        ChangeDirAction( DirectionWidget& plugin, const std::shared_ptr<Object>& obj ) :
            ChangeXfAction( "Change Dir", obj ),
            plugin_{ plugin },
            dir_{ plugin.dir_ }
        {}
        virtual void action( Type type ) override
        {
            ChangeXfAction::action( type );
            std::swap( dir_, plugin_.dir_ );

        }
    private:
        DirectionWidget& plugin_;
        Vector3f dir_;
    };

    void clear_();

public:

    MRVIEWER_API DirectionWidget( const Vector3f& dir, const Vector3f& base, float length );
    MRVIEWER_API ~DirectionWidget();
    MRVIEWER_API void updateDirection( const Vector3f& dir );
    MRVIEWER_API void updateArrow( const Vector3f& base, float length );
};

}
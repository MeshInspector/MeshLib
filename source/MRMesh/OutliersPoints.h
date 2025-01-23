#pragma once
#include "MRMeshFwd.h"

namespace MR
{

class DetectOutliersPoints
{
public:
    void init( std::shared_ptr<PointCloud> );
    void preprocess( ProgressCallback ); // prepareCachedData
    void setRadius( float );

    class OutliersType
    {
    public:
        OutliersType( bool available ) : available_( available )
        {}
        void calculate( const VertBitSet&, ProgressCallback ) = 0; // prepareToUpdatePoints

        void setActive( bool active );

    private:
        bool active_ = true;
        bool available_ = true;

        VertBitSet resultVerts_;
        VertBitSet preparedVerts_;
    };

    class SmallComponents : public OutliersType
    {
    public:
        SmallComponents( bool available ) : OutliersType( available )
        {}

        void calculate();

        struct Settings
        {
            int maxClasterSize = 0;
        };
        void setSettings( const Settings& );
        const Settings& getSettings();

    private:
        Settings settings_;
    };

private:
    float radius_ = 0.f;

};

}

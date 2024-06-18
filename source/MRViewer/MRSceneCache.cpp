#include "MRSceneCache.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRTimer.h"

#include <iostream>

namespace MR
{

void SceneCache::invalidateAll()
{
    instance_().cachedData_.clear();
}

MR::SceneCache& SceneCache::instance_()
{
    static SceneCache sceneCahce;
    return sceneCahce;
}

}

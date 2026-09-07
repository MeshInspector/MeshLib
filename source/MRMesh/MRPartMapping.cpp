#include "MRPartMapping.h"
#include "MRMapOrHashMap.h"
#include "MRId.h"

namespace MR
{

void PartMapping::clear()
{
    if ( src2tgtFaces )
        src2tgtFaces->clear();
    if ( src2tgtVerts )
        src2tgtVerts->clear();
    if ( src2tgtEdges )
        src2tgtEdges->clear();
    if ( tgt2srcFaces )
        tgt2srcFaces->clear();
    if ( tgt2srcVerts )
        tgt2srcVerts->clear();
    if ( tgt2srcEdges )
        tgt2srcEdges->clear();
}

} //namespace MR

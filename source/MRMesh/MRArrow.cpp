#include "MRArrow.h"
#include "MRMesh.h"
#include "MRMatrix3.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"

namespace MR
{

Mesh makeArrow(const Vector3f& base, const Vector3f& vert, const float& thickness, const float& coneRadius, const float coneSize, const int qual)
{
    // create object Mesh
    Mesh meshObj;

    //topology
    std::vector<VertId> v;
    v.reserve(qual * 18ull);
    //first 2 points are base and vert
    for (int i = 0; i < qual; i++)
    {
        // base tri
        v.emplace_back(VertId{ 0 });
        v.emplace_back(VertId{ 3 * ((i + 1) % qual) + 2 });
        v.emplace_back(VertId{ 3 * i + 2 });
        // cap tri
        v.emplace_back(VertId{ 1 });
        v.emplace_back(VertId{ 3 * i + 4 });
        v.emplace_back(VertId{ 3 * ((i + 1) % qual) + 4 });

        // length
        v.emplace_back(VertId{ 3 * i + 2 });
        v.emplace_back(VertId{ 3 * ((i + 1) % qual) + 3 });
        v.emplace_back(VertId{ 3 * i + 3 });
        v.emplace_back(VertId{ 3 * ((i + 1) % qual) + 3 });
        v.emplace_back(VertId{ 3 * i + 2 });
        v.emplace_back(VertId{ 3 * ((i + 1) % qual) + 2 });

        // under
        v.emplace_back(VertId{ 3 * i + 3 });
        v.emplace_back(VertId{ 3 * ((i + 1) % qual) + 4 });
        v.emplace_back(VertId{ 3 * i + 4 });
        v.emplace_back(VertId{ 3 * ((i + 1) % qual) + 4 });
        v.emplace_back(VertId{ 3 * i + 3 });
        v.emplace_back(VertId{ 3 * ((i + 1) % qual) + 3 });
    }
    meshObj.topology = MeshBuilder::fromVertexTriples(v);

    //geometry
    auto& p = meshObj.points;
    p.reserve(qual*3ull + 2);
    p.emplace_back(base);
    p.emplace_back(vert);

    const Vector3f coneAxis = (vert - base).normalized();
    const Vector3f mid = vert - coneSize * coneAxis;
    const Vector3f furAxis = coneAxis.furthestBasisVector();
    const Vector3f baseRad = cross(coneAxis, furAxis).normalized() * thickness;
    const Vector3f coneRad = cross(coneAxis, furAxis).normalized() * coneRadius;
    float step = 2.0f * PI_F / qual;
    for (int i = 0; i < qual; i++)
    {
        p.emplace_back(base + Matrix3f::rotation(coneAxis, step * i)* baseRad);
        p.emplace_back(mid + Matrix3f::rotation(coneAxis, step * i) * baseRad);
        p.emplace_back(mid + Matrix3f::rotation(coneAxis, step * i) * coneRad);
    }

    return meshObj;
}

Mesh makeBasisAxes(const float& size, const float& thickness, const float& coneRadius, const float coneSize, const int qual)
{
    Vector3f base(0.0f, 0.0f, 0.0f);
    Mesh meshObjX = makeArrow(base, base + Vector3f::plusX() * size, thickness, coneRadius, coneSize, qual);
    Mesh meshObjY = makeArrow(base, base + Vector3f::plusY() * size, thickness, coneRadius, coneSize, qual);
    Mesh meshObjZ = makeArrow(base, base + Vector3f::plusZ() * size, thickness, coneRadius, coneSize, qual);
    meshObjX.addPart(meshObjY);
    meshObjX.addPart(meshObjZ);
    return meshObjX;
}

}

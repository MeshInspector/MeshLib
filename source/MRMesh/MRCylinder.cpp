#include "MRCylinder.h"
#include "MRMesh.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"

namespace MR
{

MR::Mesh makeCylinder(float radius, float length,
    int resolution)
{
    std::vector<Vector3f> points(2 * resolution + 2);
    auto step = 2.0f * PI_F / float(resolution);

    for (int h = 0; h < 2; ++h)
    {
        points[h] = Vector3f(0.0f, 0.0f, float(h)*length);
        for (int anglei = 0; anglei < resolution; ++anglei)
        {
            float angle = anglei * step;
            auto ind = anglei + resolution * h + 2;
            points[ind].x = radius * cosf(angle);
            points[ind].y = radius * sinf(angle);
            points[ind].z = h * length;
        }
    }

    std::vector<MeshBuilder::Triangle> tris;
    tris.reserve(4 * resolution);
    for (int i = 0; i < resolution; ++i)
    {
        tris.emplace_back(VertId(0), VertId(((i + 1) % resolution) + 2),
            VertId(i + 2), FaceId(4 * i));
        tris.emplace_back(VertId(1), VertId(i + 2 + resolution),
            VertId(((i + 1) % resolution) + 2 + resolution), FaceId(4 * i + 1));
        tris.emplace_back(VertId(i + 2), VertId(((i + 1) % resolution) + 2),
            VertId(i + 2 + resolution), FaceId(4 * i + 2));
        tris.emplace_back(VertId(((i + 1) % resolution) + 2), VertId(((i + 1) % resolution) + 2 + resolution),
            VertId(i + 2 + resolution), FaceId(4 * i + 3));
    }
    Mesh res;
    res.topology = MeshBuilder::fromTriangles(tris);
    res.points.vec_ = std::move(points);

    return res;
}

MR::Mesh makeCylinderAdvanced( float radius0, float radius1, float start_angle,
                               float arc_size, float length, int resolution )
{
    int cap0 = 0, cap1 = 0;
    bool slice = false;

    if (radius0 != 0)
    {
        cap0 = resolution;
    }

    if (radius1 != 0)
    {
        cap1 = resolution;
    }

    if (arc_size >= 2.0f * PI_F)
        arc_size = 2.0f * PI_F;
    else if (arc_size <= -2.0f * PI_F)
        arc_size = -2.0f * PI_F;
    else
    {
        if (cap0) ++cap0;
        if (cap1) ++cap1;
        slice = true;
    }

    std::vector<Vector3f> points(cap0 + cap1 + 2);
    auto step = arc_size / float(resolution);
    int ver_id = 0;

    points[ver_id++] = Vector3f(0.0f, 0.0f, 0.0f);
    points[ver_id++] = Vector3f(0.0f, 0.0f, length);

    for (int anglei = 0; anglei < cap0; ++anglei)
    {
        float angle = start_angle + anglei * step;
        points[ver_id].x = radius0 * cosf(angle);
        points[ver_id].y = radius0 * sinf(angle);
        points[ver_id].z = 0;
        ++ver_id;
    }

    for (int anglei = 0; anglei < cap1; ++anglei)
    {
        float angle = start_angle + anglei * step;
        points[ver_id].x = radius1 * cosf(angle);
        points[ver_id].y = radius1 * sinf(angle);
        points[ver_id].z = length;
        ++ver_id;
    }

    std::vector<MeshBuilder::Triangle> tris;
    tris.reserve(2 * (cap0 + cap1) + 4 * slice);
    int face_id = 0;

    for (int i = 0; i < resolution; ++i)
    {
        if (cap0)
        {
            tris.emplace_back(VertId(0), VertId(((i + 1) % cap0) + 2),
                VertId(i + 2), FaceId(face_id++));
            tris.emplace_back(VertId(i + 2), VertId(((i + 1) % cap0) + 2),
                VertId(cap1 ? i + 2 + cap0 : 1), FaceId(face_id++));
        }
        if (cap1)
        {
            tris.emplace_back(VertId(1), VertId(i + 2 + cap0),
                VertId(((i + 1) % cap1) + 2 + cap0), FaceId(face_id++));
            if (cap0)
            {
                tris.emplace_back(VertId(((i + 1) % cap1) + 2), VertId(((i + 1) % cap1) + 2 + cap0),
                    VertId(i + 2 + cap0), FaceId(face_id++));
            }
            else
            {
                tris.emplace_back(VertId(0), VertId(((i + 1) % cap1) + 2),
                    VertId(i + 2 + cap0), FaceId(face_id++));
            }
        }
    }
    if (slice)
    {
        if (cap0)
        {
            tris.emplace_back(VertId(0), VertId(2),
                VertId(cap1 ? 2 + cap0 : 1), FaceId(face_id++));
            tris.emplace_back(VertId(resolution + 2), VertId(0),
                VertId(cap1 ? resolution + 2 + cap0 : 1), FaceId(face_id++));
        }
        if (cap1)
        {
            tris.emplace_back(VertId(1), VertId(0),
                VertId(2 + cap0), FaceId(face_id++));
            tris.emplace_back(VertId(0), VertId(1),
                VertId(resolution + 2 + cap0), FaceId(face_id++));
        }
    }

    Mesh res;
    res.topology = MeshBuilder::fromTriangles(tris);
    res.points.vec_ = std::move(points);

    return res;
}

}

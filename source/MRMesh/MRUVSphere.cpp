#include "MRUVSphere.h"
#include "MRMesh.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"

namespace MR
{
    MR::Mesh makeUVSphere(float radius, int horisontalResolution, int verticalResolution) 
    {
        MR_TIMER;

        int top_cap, bottom_cap, i, j, k;
        float a, b;

        int vertexCount = horisontalResolution * verticalResolution + 2;

        std::vector<Vector3f> points(vertexCount);
        
        k = 0;
        for (j = 0; j < verticalResolution; ++j) {
            a = PI_F * ((float)(j + 1) / (verticalResolution + 1) - 0.5f);
            for (i = 0; i < horisontalResolution; ++i) {
                b = 2 * i * PI_F / horisontalResolution;

                points[k].x = (float)(radius * cos(a) * sin(b));
                points[k].y = (float)(radius * cos(a) * cos(b));
                points[k].z = (float)(radius * sin(a));
                ++k;
            }
        }

        points[k].x = 0;
        points[k].y = 0;
        points[k].z = -radius;
        bottom_cap = k;
        ++k;

        points[k].x = 0;
        points[k].y = 0;
        points[k].z = radius;
        top_cap = k;

        std::vector<MeshBuilder::Triangle> tris;

        int triangleCount = 2 * horisontalResolution * verticalResolution;
        tris.reserve(triangleCount);

        k = 0;
        for (j = 0; j < verticalResolution - 1; ++j) {
            for (i = 0; i < horisontalResolution; ++i) {
                tris.emplace_back(
                    VertId((j + 1) * horisontalResolution + i),
                    VertId(j * horisontalResolution + (i + 1) % horisontalResolution),
                    VertId(j * horisontalResolution + i),
                    FaceId(k)); ++k;

                tris.emplace_back(
                    VertId((j + 1) * horisontalResolution + i),
                    VertId((j + 1) * horisontalResolution + (i + 1) % horisontalResolution),
                    VertId(j * horisontalResolution + (i + 1) % horisontalResolution),
                    FaceId(k)); ++k;
            }
        }

        for (i = 0; i < horisontalResolution; ++i) {
            tris.emplace_back(
                VertId(0 * horisontalResolution + i),
                VertId(0 * horisontalResolution + (i + 1) % horisontalResolution),
                VertId(bottom_cap),
                FaceId(k)); ++k;

            tris.emplace_back(
                VertId((verticalResolution - 1) * horisontalResolution + (i + 1) % horisontalResolution),
                VertId((verticalResolution - 1) * horisontalResolution + i),
                VertId(top_cap),
                FaceId(k)); ++k;
        }

        Mesh res;
        res.topology = MeshBuilder::fromTriangles(tris);
        res.points.vec_ = std::move(points);

        return res;
    }
}

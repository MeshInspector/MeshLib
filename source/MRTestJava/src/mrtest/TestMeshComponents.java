package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import meshlib.mr.BooleanOperation;
import meshlib.mr.BooleanResult;
import meshlib.mr.FaceBitSet;
import meshlib.mr.FaceId;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.MeshPart;
import meshlib.mr.Vector3f;
import meshlib.mr.std_pair_MR_FaceBitSet_int;
import meshlib.mr.std_pair_MR_Face2RegionMap_int;

public final class TestMeshComponents
{
    static MeshPart createMesh()
    {
        Mesh meshA = Functions.makeCube(Vector3f.diagonal(1.0f), Vector3f.diagonal(-0.5f));
        Mesh meshB = Functions.makeCube(Vector3f.diagonal(0.1f), Vector3f.diagonal(1.0f));
        BooleanResult result = Functions.boolean_4_const_MR_Mesh_ref(meshA, meshB, BooleanOperation.Union, null);
        Mesh mesh = new Mesh(result.mesh());
        return new MeshPart(mesh, null); // The part keeps the mesh alive.
    }

    public static void componentsMap()
    {
        MeshPart m = createMesh();
        std_pair_MR_Face2RegionMap_int map = meshlib.mr.meshcomponents.Functions.getAllComponentsMap(m, null, null);
        checkEq(map.second(), 2, "components");
        checkEq(map.first().size(), 24, "map size");
        checkEq(map.first().index(new FaceId(0)).id_(), 0, "face 0");
        checkEq(map.first().index(new FaceId(12)).id_(), 1, "face 12");
    }

    public static void largeRegions()
    {
        MeshPart m = createMesh();
        std_pair_MR_Face2RegionMap_int map = meshlib.mr.meshcomponents.Functions.getAllComponentsMap(m, null, null);
        std_pair_MR_FaceBitSet_int regions = meshlib.mr.meshcomponents.Functions.getLargeByAreaRegions(m, map.first(), map.second(), 0.1f);
        checkEq(regions.second(), 1, "regions");
        check(regions.first().test(new FaceId(0)), "face 0");
        check(!regions.first().test(new FaceId(12)), "face 12");
    }

    public static void largeComponents()
    {
        MeshPart m = createMesh();
        FaceBitSet components = meshlib.mr.meshcomponents.Functions.getLargeByAreaComponents(m, 0.1f, null);
        check(components.test(new FaceId(0)), "face 0");
        check(!components.test(new FaceId(12)), "face 12");
    }

    public static void largestComponent()
    {
        MeshPart m = createMesh();
        try (Arena arena = Arena.ofConfined())
        {
            // The `int *` output parameter is raw.
            MemorySegment smallerComponents = arena.allocate(ValueLayout.JAVA_INT);
            FaceBitSet largest = meshlib.mr.meshcomponents.Functions.getLargestComponent(m, null, null, 0.1f, smallerComponents);
            check(largest.test(new FaceId(0)), "face 0");
            check(!largest.test(new FaceId(12)), "face 12");
            checkEq(smallerComponents.get(ValueLayout.JAVA_INT, 0), 1, "smaller components");
        }
    }

    public static void getComponent()
    {
        MeshPart m = createMesh();
        FaceBitSet component = meshlib.mr.meshcomponents.Functions.getComponent(m, new FaceId(12), null, null);
        check(component.test(new FaceId(12)), "face 12");
        check(!component.test(new FaceId(0)), "face 0");
    }
}

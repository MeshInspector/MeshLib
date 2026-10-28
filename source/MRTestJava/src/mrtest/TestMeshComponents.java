package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.BooleanOperation;
import meshlib.BooleanParameters;
import meshlib.BooleanResult;
import meshlib.FaceBitSet;
import meshlib.FaceId;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.Vector3f;
import meshlib.int_output;
import meshlib.std_pair_MR_FaceBitSet_int;
import meshlib.std_pair_MR_Face2RegionMap_int;

public final class TestMeshComponents
{
    static MeshPart createMesh()
    {
        Mesh meshA = Functions.makeCube(Vector3f.diagonal(1.0f), Vector3f.diagonal(-0.5f));
        Mesh meshB = Functions.makeCube(Vector3f.diagonal(0.1f), Vector3f.diagonal(1.0f));
        BooleanResult result = Functions.boolean_(meshA, meshB, BooleanOperation.Union, new BooleanParameters());
        Mesh mesh = new Mesh(result.mesh());
        // MeshPart(Mesh, FaceBitSet) does not keep the mesh alive, so the part owns it here
        return new MeshPart(new MeshPart(mesh, null).release(), true, mesh);
    }

    public static void componentsMap()
    {
        MeshPart m = createMesh();
        std_pair_MR_Face2RegionMap_int map = meshlib.meshcomponents.Functions.getAllComponentsMap(m, null, null);
        checkEq(map.second(), 2, "components");
        checkEq(map.first().size(), 24, "map size");
        checkEq(map.first().index(new FaceId(0)).id_(), 0, "face 0");
        checkEq(map.first().index(new FaceId(12)).id_(), 1, "face 12");
    }

    public static void largeRegions()
    {
        MeshPart m = createMesh();
        std_pair_MR_Face2RegionMap_int map = meshlib.meshcomponents.Functions.getAllComponentsMap(m, null, null);
        std_pair_MR_FaceBitSet_int regions = meshlib.meshcomponents.Functions.getLargeByAreaRegions(m, map.first(), map.second(), 0.1f);
        checkEq(regions.second(), 1, "regions");
        check(regions.first().test(new FaceId(0)), "face 0");
        check(!regions.first().test(new FaceId(12)), "face 12");
    }

    public static void largeComponents()
    {
        MeshPart m = createMesh();
        FaceBitSet components = meshlib.meshcomponents.Functions.getLargeByAreaComponents(m, 0.1f, null);
        check(components.test(new FaceId(0)), "face 0");
        check(!components.test(new FaceId(12)), "face 12");
    }

    public static void largestComponent()
    {
        MeshPart m = createMesh();
        // The `int *` output parameter takes an `int_output`.
        int_output smallerComponents = new int_output();
        FaceBitSet largest = meshlib.meshcomponents.Functions.getLargestComponent(m, null, null, 0.1f, smallerComponents);
        check(largest.test(new FaceId(0)), "face 0");
        check(!largest.test(new FaceId(12)), "face 12");
        checkEq(smallerComponents.value(), 1, "smaller components");
    }

    public static void getComponent()
    {
        MeshPart m = createMesh();
        FaceBitSet component = meshlib.meshcomponents.Functions.getComponent(m, new FaceId(12), null, null);
        check(component.test(new FaceId(12)), "face 12");
        check(!component.test(new FaceId(0)), "face 0");
    }
}

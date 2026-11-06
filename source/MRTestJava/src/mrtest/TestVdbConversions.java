package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.Box3f;
import meshlib.Box3i;
import meshlib.FloatGrid;
import meshlib.Functions;
import meshlib.GridToMeshSettings;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.MeshToVolumeParams;
import meshlib.SphereParams;
import meshlib.VdbVolume;
import meshlib.Vector3f;
import meshlib.Vector3i;
import meshlib.VoxelBitSet;
import meshlib.VoxelId;

public final class TestVdbConversions
{
    static VdbVolume createVolume()
    {
        SphereParams sphereParams = new SphereParams();
        sphereParams.setRadius(1.0f);
        sphereParams.setNumMeshVertices(3000);
        Mesh mesh = Functions.makeSphere(sphereParams);
        MeshPart mp = new MeshPart(mesh, null);
        MeshToVolumeParams params = new MeshToVolumeParams();
        params.setVoxelSize(Vector3f.diagonal(0.1f));
        return Functions.meshToVolume(mp, params);
    }

    static boolean dims(Vector3i dims, int x, int y, int z)
    {
        return dims.x() == x && dims.y() == y && dims.z() == z;
    }

    public static void vdbConversions()
    {
        VdbVolume volume = createVolume();
        check(volume.min() > -0.001f && volume.min() < 0.001f, "min");
        check(volume.max() > 2.999f && volume.max() < 3.001f, "max");
        check(dims(volume.dims(), 26, 26, 26), "dims");

        GridToMeshSettings settings = new GridToMeshSettings();
        settings.setVoxelSize(Vector3f.diagonal(0.1f));
        settings.setIsoValue(1.0f);
        Mesh restored = Functions.gridToMesh_const_MR_FloatGrid_ref(volume.data(), settings);
        Box3f bbox = restored.computeBoundingBox((meshlib.AffineXf3f)null);
        check(bbox.min().x() > 0.199f && bbox.min().x() < 0.201f, "min x");
        check(bbox.min().y() > 0.199f && bbox.min().y() < 0.201f, "min y");
        check(bbox.min().z() > 0.199f && bbox.min().z() < 0.201f, "min z");
        check(bbox.max().x() > 2.394f && bbox.max().x() < 2.396f, "max x");
        check(bbox.max().y() > 2.394f && bbox.max().y() < 2.396f, "max y");
        check(bbox.max().z() > 2.394f && bbox.max().z() < 2.396f, "max z");
        checkEq(restored.points().size(), 3748, "points");
    }

    public static void uniformResampling()
    {
        VdbVolume volume = createVolume();
        FloatGrid resampledGrid = Functions.resampled(volume.data(), 2.0f, null);
        VdbVolume resampledVolume = Functions.floatGridToVdbVolume(resampledGrid);
        check(dims(resampledVolume.dims(), 13, 13, 13), "dims");
    }

    public static void resampling()
    {
        VdbVolume volume = createVolume();
        FloatGrid resampledGrid = Functions.resampled(volume.data(), new Vector3f(2.0f, 1.0f, 0.5f), null);
        VdbVolume resampledVolume = Functions.floatGridToVdbVolume(resampledGrid);
        check(dims(resampledVolume.dims(), 13, 27, 53), "dims");
    }

    public static void cropping()
    {
        VdbVolume volume = createVolume();
        Box3i box = new Box3i(new Vector3i(2, 5, 1), new Vector3i(18, 13, 23));
        FloatGrid croppedGrid = Functions.cropped(volume.data(), box, null);
        VdbVolume croppedVolume = Functions.floatGridToVdbVolume(croppedGrid);
        check(dims(croppedVolume.dims(), 16, 8, 22), "dims");
    }

    public static void accessors()
    {
        VdbVolume volume = createVolume();
        Vector3i p = new Vector3i(0, 0, 0);
        check(Functions.getValue(volume.data(), p) == 3.0f, "initial value");
        Vector3i dims = volume.dims();
        VoxelBitSet region = new VoxelBitSet();
        region.resize((long)dims.x() * dims.y() * dims.z(), null);
        region.set(new VoxelId(0), true);
        Functions.setValue(volume.data(), region, 1.0f);
        check(Functions.getValue(volume.data(), p) == 1.0f, "value after region set");
        Functions.setValue(volume.data(), p, 2.0f);
        check(Functions.getValue(volume.data(), p) == 2.0f, "value after point set");
    }
}

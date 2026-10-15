package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.Box3f;
import meshlib.mr.Box3i;
import meshlib.mr.FloatGrid;
import meshlib.mr.Functions;
import meshlib.mr.GridToMeshSettings;
import meshlib.mr.Mesh;
import meshlib.mr.MeshPart;
import meshlib.mr.MeshToVolumeParams;
import meshlib.mr.SphereParams;
import meshlib.mr.VdbVolume;
import meshlib.mr.Vector3f;
import meshlib.mr.Vector3i;
import meshlib.mr.VoxelBitSet;
import meshlib.mr.VoxelId;
import meshlib.mr.expected_MR_Mesh_std_string;
import meshlib.mr.expected_MR_VdbVolume_std_string;

public final class TestVdbConversions
{
    static expected_MR_VdbVolume_std_string createVolume()
    {
        SphereParams sphereParams = new SphereParams();
        sphereParams.setRadius(1.0f);
        sphereParams.setNumMeshVertices(3000);
        Mesh mesh = Functions.makeSphere(sphereParams);
        MeshPart mp = new MeshPart(mesh, null);
        MeshToVolumeParams params = new MeshToVolumeParams();
        params.setVoxelSize(Vector3f.diagonal(0.1f));
        expected_MR_VdbVolume_std_string volume = Functions.meshToVolume(mp, params);
        check(volume.success(), "meshToVolume failed");
        return volume;
    }

    static boolean dims(Vector3i dims, int x, int y, int z)
    {
        return dims.x() == x && dims.y() == y && dims.z() == z;
    }

    public static void vdbConversions()
    {
        expected_MR_VdbVolume_std_string volumeHolder = createVolume();
        VdbVolume volume = volumeHolder.value_mut();
        check(volume.min() > -0.001f && volume.min() < 0.001f, "min");
        check(volume.max() > 2.999f && volume.max() < 3.001f, "max");
        check(dims(volume.dims(), 26, 26, 26), "dims");

        GridToMeshSettings settings = new GridToMeshSettings();
        settings.setVoxelSize(Vector3f.diagonal(0.1f));
        settings.setIsoValue(1.0f);
        expected_MR_Mesh_std_string restored = Functions.gridToMesh_const_MR_FloatGrid_ref(volume.data(), settings);
        check(restored.success(), "gridToMesh failed");
        Box3f bbox = restored.value().computeBoundingBox((meshlib.mr.AffineXf3f)null);
        check(bbox.min().x() > 0.199f && bbox.min().x() < 0.201f, "min x");
        check(bbox.min().y() > 0.199f && bbox.min().y() < 0.201f, "min y");
        check(bbox.min().z() > 0.199f && bbox.min().z() < 0.201f, "min z");
        check(bbox.max().x() > 2.394f && bbox.max().x() < 2.396f, "max x");
        check(bbox.max().y() > 2.394f && bbox.max().y() < 2.396f, "max y");
        check(bbox.max().z() > 2.394f && bbox.max().z() < 2.396f, "max z");
        checkEq(restored.value().points().size(), 3748, "points");
    }

    public static void uniformResampling()
    {
        expected_MR_VdbVolume_std_string volumeHolder = createVolume();
        VdbVolume volume = volumeHolder.value_mut();
        FloatGrid resampledGrid = Functions.resampled(volume.data(), 2.0f, null);
        VdbVolume resampledVolume = Functions.floatGridToVdbVolume(resampledGrid);
        check(dims(resampledVolume.dims(), 13, 13, 13), "dims");
    }

    public static void resampling()
    {
        expected_MR_VdbVolume_std_string volumeHolder = createVolume();
        VdbVolume volume = volumeHolder.value_mut();
        FloatGrid resampledGrid = Functions.resampled(volume.data(), new Vector3f(2.0f, 1.0f, 0.5f), null);
        VdbVolume resampledVolume = Functions.floatGridToVdbVolume(resampledGrid);
        check(dims(resampledVolume.dims(), 13, 27, 53), "dims");
    }

    public static void cropping()
    {
        expected_MR_VdbVolume_std_string volumeHolder = createVolume();
        VdbVolume volume = volumeHolder.value_mut();
        Box3i box = new Box3i(new Vector3i(2, 5, 1), new Vector3i(18, 13, 23));
        FloatGrid croppedGrid = Functions.cropped(volume.data(), box, null);
        VdbVolume croppedVolume = Functions.floatGridToVdbVolume(croppedGrid);
        check(dims(croppedVolume.dims(), 16, 8, 22), "dims");
    }

    public static void accessors()
    {
        expected_MR_VdbVolume_std_string volumeHolder = createVolume();
        VdbVolume volume = volumeHolder.value_mut();
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

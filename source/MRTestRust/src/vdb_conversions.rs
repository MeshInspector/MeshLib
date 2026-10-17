use meshlib::{Box3i, Vector3f, Vector3i, VoxelId};

fn create_volume() -> meshlib::Owned<meshlib::expected_MR_VdbVolume_std_string>
{
    let mut sphere_params = meshlib::SphereParams::new();
    sphere_params.set_radius(1.0);
    sphere_params.set_num_mesh_vertices(3000);
    let mesh = meshlib::make_sphere(&sphere_params);
    let mp = meshlib::MeshPart::construct(&mesh, None);
    let mut params = meshlib::MeshToVolumeParams::new();
    params.set_voxel_size(Vector3f::diagonal(0.1));
    let volume = meshlib::mesh_to_volume(&mp, Some(&params));
    assert!(volume.success(), "meshToVolume failed");
    volume
}

#[test]
fn vdb_conversions()
{
    let mut volume = create_volume();
    let volume = volume.value_mut().unwrap();
    assert!(*volume.min() > -0.001 && *volume.min() < 0.001);
    assert!(*volume.max() > 2.999 && *volume.max() < 3.001);
    let dims = volume.dims();
    assert_eq!((dims.x, dims.y, dims.z), (26, 26, 26));

    let mut settings = meshlib::GridToMeshSettings::new();
    settings.set_voxel_size(Vector3f::diagonal(0.1));
    settings.set_iso_value(1.0);
    let restored = meshlib::grid_to_mesh_const_mr_float_grid_ref(volume.data(), &settings);
    let restored = restored.into_result().expect("gridToMesh failed");
    let bbox = restored.compute_bounding_box(None);
    assert!(bbox.min.x > 0.199 && bbox.min.x < 0.201);
    assert!(bbox.min.y > 0.199 && bbox.min.y < 0.201);
    assert!(bbox.min.z > 0.199 && bbox.min.z < 0.201);
    assert!(bbox.max.x > 2.394 && bbox.max.x < 2.396);
    assert!(bbox.max.y > 2.394 && bbox.max.y < 2.396);
    assert!(bbox.max.z > 2.394 && bbox.max.z < 2.396);
    assert_eq!(restored.points().size(), 3748);
}

#[test]
fn uniform_resampling()
{
    let mut volume = create_volume();
    let volume = volume.value_mut().unwrap();
    let resampled_grid = meshlib::resampled_float(volume.data_mut(), 2.0, None);
    let resampled_volume = meshlib::float_grid_to_vdb_volume(&resampled_grid);
    let dims = resampled_volume.dims();
    assert_eq!((dims.x, dims.y, dims.z), (13, 13, 13));
}

#[test]
fn resampling()
{
    let mut volume = create_volume();
    let volume = volume.value_mut().unwrap();
    let resampled_grid = meshlib::resampled_mr_vector3f(volume.data_mut(), &Vector3f {x: 2.0, y: 1.0, z: 0.5}, None);
    let resampled_volume = meshlib::float_grid_to_vdb_volume(&resampled_grid);
    let dims = resampled_volume.dims();
    assert_eq!((dims.x, dims.y, dims.z), (13, 27, 53));
}

#[test]
fn cropping()
{
    let mut volume = create_volume();
    let volume = volume.value_mut().unwrap();
    let b = Box3i {min: Vector3i {x: 2, y: 5, z: 1}, max: Vector3i {x: 18, y: 13, z: 23}};
    let cropped_grid = meshlib::cropped(volume.data_mut(), &b, None);
    let cropped_volume = meshlib::float_grid_to_vdb_volume(&cropped_grid);
    let dims = cropped_volume.dims();
    assert_eq!((dims.x, dims.y, dims.z), (16, 8, 22));
}

#[test]
fn accessors()
{
    let mut volume = create_volume();
    let volume = volume.value_mut().unwrap();
    let p = Vector3i {x: 0, y: 0, z: 0};
    assert_eq!(meshlib::get_value(volume.data_mut(), &p), 3.0);
    let dims = *volume.dims();
    let mut region = meshlib::VoxelBitSet::new();
    region.resize((dims.x * dims.y * dims.z) as _, None);
    region.set_2(VoxelId {id_: 0}, true);
    meshlib::set_value_mr_voxel_bit_set(volume.data_mut(), &region, 1.0);
    assert_eq!(meshlib::get_value(volume.data_mut(), &p), 1.0);
    meshlib::set_value_mr_vector3i(volume.data_mut(), &p, 2.0);
    assert_eq!(meshlib::get_value(volume.data_mut(), &p), 2.0);
}

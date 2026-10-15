use crate::common::create_cube;

#[test]
fn offset_mesh()
{
    let mesh = create_cube();
    let mp = meshlib::MeshPart::construct(&mesh, None);
    let mut params = meshlib::OffsetParameters::new();
    params.set_voxel_size(meshlib::suggest_voxel_size(&mp, 10000.0));
    assert!(meshlib::offset_mesh(&mp, 0.1, Some(&params)).success());
}

#[test]
fn double_offset_mesh()
{
    let mesh = create_cube();
    let mp = meshlib::MeshPart::construct(&mesh, None);
    let mut params = meshlib::OffsetParameters::new();
    params.set_voxel_size(meshlib::suggest_voxel_size(&mp, 10000.0));
    assert!(meshlib::double_offset_mesh(&mp, 0.1, -0.2, Some(&params)).success());
}

#[test]
fn mc_offset_mesh()
{
    let mesh = create_cube();
    let mp = meshlib::MeshPart::construct(&mesh, None);
    let mut params = meshlib::OffsetParameters::new();
    params.set_voxel_size(meshlib::suggest_voxel_size(&mp, 10000.0));
    assert!(meshlib::mc_offset_mesh(&mp, 0.1, Some(&params), None).success());
}

#[test]
fn sharp_offset_mesh()
{
    let mesh = create_cube();
    let mp = meshlib::MeshPart::construct(&mesh, None);
    let mut params = meshlib::SharpOffsetParameters::new();
    params.set_voxel_size(meshlib::suggest_voxel_size(&mp, 10000.0));
    assert!(meshlib::sharp_offset_mesh(&mp, 0.1, Some(&params)).success());
}

#[test]
fn general_offset_mesh()
{
    let mesh = create_cube();
    let mp = meshlib::MeshPart::construct(&mesh, None);
    let mut params = meshlib::GeneralOffsetParameters::new();
    params.set_voxel_size(meshlib::suggest_voxel_size(&mp, 10000.0));
    params.set_mode(meshlib::OffsetMode::Standard);
    assert!(meshlib::general_offset_mesh(&mp, 0.1, &params).success());
}

#[test]
fn thicken_mesh()
{
    let mesh = create_cube();
    let mp = meshlib::MeshPart::construct(&mesh, None);
    let mut params = meshlib::GeneralOffsetParameters::new();
    params.set_voxel_size(meshlib::suggest_voxel_size(&mp, 10000.0));
    params.set_mode(meshlib::OffsetMode::Standard);
    let map = meshlib::PartMapping::new();
    assert!(meshlib::thicken_mesh(&mesh, 0.1, Some(&params), Some(&map)).success());
}

use crate::common::create_cube;
use meshlib::mr;

#[test]
fn offset_mesh()
{
    let mesh = create_cube();
    let mp = mr::MeshPart::construct(&mesh, None);
    let mut params = mr::OffsetParameters::new();
    params.set_voxel_size(mr::suggest_voxel_size(&mp, 10000.0));
    assert!(mr::offset_mesh(&mp, 0.1, Some(&params)).success());
}

#[test]
fn double_offset_mesh()
{
    let mesh = create_cube();
    let mp = mr::MeshPart::construct(&mesh, None);
    let mut params = mr::OffsetParameters::new();
    params.set_voxel_size(mr::suggest_voxel_size(&mp, 10000.0));
    assert!(mr::double_offset_mesh(&mp, 0.1, -0.2, Some(&params)).success());
}

#[test]
fn mc_offset_mesh()
{
    let mesh = create_cube();
    let mp = mr::MeshPart::construct(&mesh, None);
    let mut params = mr::OffsetParameters::new();
    params.set_voxel_size(mr::suggest_voxel_size(&mp, 10000.0));
    assert!(mr::mc_offset_mesh(&mp, 0.1, Some(&params), None).success());
}

#[test]
fn sharp_offset_mesh()
{
    let mesh = create_cube();
    let mp = mr::MeshPart::construct(&mesh, None);
    let mut params = mr::SharpOffsetParameters::new();
    params.set_voxel_size(mr::suggest_voxel_size(&mp, 10000.0));
    assert!(mr::sharp_offset_mesh(&mp, 0.1, Some(&params)).success());
}

#[test]
fn general_offset_mesh()
{
    let mesh = create_cube();
    let mp = mr::MeshPart::construct(&mesh, None);
    let mut params = mr::GeneralOffsetParameters::new();
    params.set_voxel_size(mr::suggest_voxel_size(&mp, 10000.0));
    params.set_mode(mr::OffsetMode::Standard);
    assert!(mr::general_offset_mesh(&mp, 0.1, &params).success());
}

#[test]
fn thicken_mesh()
{
    let mesh = create_cube();
    let mp = mr::MeshPart::construct(&mesh, None);
    let mut params = mr::GeneralOffsetParameters::new();
    params.set_voxel_size(mr::suggest_voxel_size(&mp, 10000.0));
    params.set_mode(mr::OffsetMode::Standard);
    let map = mr::PartMapping::new();
    assert!(mr::thicken_mesh(&mesh, 0.1, Some(&params), Some(&map)).success());
}

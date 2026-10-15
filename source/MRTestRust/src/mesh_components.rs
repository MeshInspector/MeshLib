use meshlib::mr;
use meshlib::mr::{mesh_components, FaceId, Vector3f};

struct CreatedMesh
{
    // The part refers to the mesh, so the mesh must outlive it: fields drop in declaration order.
    part: meshlib::Owned<mr::MeshPart>,
    _mesh: meshlib::Owned<mr::Mesh>,
}

fn create_mesh() -> CreatedMesh
{
    let mesh_a = mr::make_cube(Some(&Vector3f::diagonal(1.0)), Some(&Vector3f::diagonal(-0.5)));
    let mesh_b = mr::make_cube(Some(&Vector3f::diagonal(0.1)), Some(&Vector3f::diagonal(1.0)));
    let mut result = mr::boolean_4_const_mr_mesh_ref(&mesh_a, &mesh_b, mr::BooleanOperation::Union, None);
    let mesh = mr::Mesh::construct_from_another(result.mesh_mut());
    let part = mr::MeshPart::construct(&mesh, None);
    CreatedMesh {part, _mesh: mesh}
}

#[test]
fn components_map()
{
    let m = create_mesh();
    let map = mesh_components::get_all_components_map(&m.part, None, None);
    assert_eq!(*map.second(), 2);
    assert_eq!(map.first().size(), 24);
    assert_eq!(map.first().index(FaceId {id_: 0}).id_, 0);
    assert_eq!(map.first().index(FaceId {id_: 12}).id_, 1);
}

#[test]
fn large_regions()
{
    let m = create_mesh();
    let map = mesh_components::get_all_components_map(&m.part, None, None);
    let regions = mesh_components::get_large_by_area_regions(&m.part, map.first(), *map.second(), 0.1);
    assert_eq!(*regions.second(), 1);
    assert!(regions.first().test(FaceId {id_: 0}));
    assert!(!regions.first().test(FaceId {id_: 12}));
}

#[test]
fn large_components()
{
    let m = create_mesh();
    let components = mesh_components::get_large_by_area_components_3(&m.part, 0.1, None);
    assert!(components.test(FaceId {id_: 0}));
    assert!(!components.test(FaceId {id_: 12}));
}

#[test]
fn largest_component()
{
    let m = create_mesh();
    let mut smaller_components = 0;
    let largest = mesh_components::get_largest_component(&m.part, None, None, Some(0.1), Some(&mut smaller_components));
    assert!(largest.test(FaceId {id_: 0}));
    assert!(!largest.test(FaceId {id_: 12}));
    assert_eq!(smaller_components, 1);
}

#[test]
fn get_component()
{
    let m = create_mesh();
    let component = mesh_components::get_component(&m.part, FaceId {id_: 12}, None, None);
    assert!(component.test(FaceId {id_: 12}));
    assert!(!component.test(FaceId {id_: 0}));
}

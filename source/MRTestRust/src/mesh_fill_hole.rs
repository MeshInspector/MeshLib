use meshlib::mr;
use meshlib::mr::{FaceId, Vector3f, VertId};

fn create_mesh_with_holes() -> meshlib::Owned<mr::Mesh>
{
    let mut points = mr::VertCoords::construct_1_size_t(6);
    *points.index_mut(VertId {id_: 0}) = Vector3f {x: 0.0, y: 0.0, z: 0.0};
    *points.index_mut(VertId {id_: 1}) = Vector3f {x: 1.0, y: 0.0, z: 0.0};
    *points.index_mut(VertId {id_: 2}) = Vector3f {x: 0.0, y: 1.0, z: 0.0};
    *points.index_mut(VertId {id_: 3}) = Vector3f {x: 0.0, y: 0.0, z: 1.0};
    *points.index_mut(VertId {id_: 4}) = Vector3f {x: 1.0, y: 0.0, z: 1.0};
    *points.index_mut(VertId {id_: 5}) = Vector3f {x: 0.0, y: 1.0, z: 1.0};
    let mut triangulation = mr::Triangulation::construct_1_size_t(5);
    let tri = |a, b, c| mr::std_array_MR_VertId_3 {elems: [VertId {id_: a}, VertId {id_: b}, VertId {id_: c}]};
    *triangulation.index_mut(FaceId {id_: 0}) = tri(0, 2, 1);
    *triangulation.index_mut(FaceId {id_: 1}) = tri(3, 4, 5);
    *triangulation.index_mut(FaceId {id_: 2}) = tri(0, 1, 3);
    *triangulation.index_mut(FaceId {id_: 3}) = tri(2, 5, 4);
    *triangulation.index_mut(FaceId {id_: 4}) = tri(2, 3, 5);
    mr::Mesh::from_triangles(&points, &triangulation, None, None)
}

#[test]
fn mesh_fill_hole()
{
    let mut mesh = create_mesh_with_holes();
    let old_bd_edges = mesh.topology().find_hole_representive_edges(None);
    assert_eq!(old_bd_edges.size(), 2);

    let mut new_faces = mr::FaceBitSet::new();
    {
        let mut params = mr::FillHoleParams::new();
        params.set_out_new_faces(Some(&mut new_faces));
        mr::fill_holes(&mut mesh, &old_bd_edges, Some(&params));
    }
    assert_eq!(new_faces.count(), 3);

    let new_bd_edges = mesh.topology().find_hole_representive_edges(None);
    assert_eq!(new_bd_edges.size(), 0);
}

#[test]
fn mesh_fill_hole_nicely()
{
    let mut mesh = create_mesh_with_holes();
    let old_bd_edges = mesh.topology().find_hole_representive_edges(None);
    assert_eq!(old_bd_edges.size(), 2);

    let settings = mr::FillHoleNicelySettings::new();
    let patch = mr::fill_hole_nicely(&mut mesh, *old_bd_edges.front().unwrap(), &settings);
    assert_eq!(patch.count(), 1887);

    let new_bd_edges = mesh.topology().find_hole_representive_edges(None);
    assert_eq!(new_bd_edges.size(), 1);
}

#[test]
fn right_boundary()
{
    let mesh = create_mesh_with_holes();
    let loops = mr::find_right_boundary(mesh.topology(), None, None);
    assert_eq!(loops.size(), 2);
    assert_eq!(loops.at(0).size(), 3);
    assert_eq!(loops.at(1).size(), 4);
}

#[test]
fn find_hole_complicating_faces()
{
    let mesh = create_mesh_with_holes();
    let faces = mr::find_hole_complicating_faces(&mesh);
    assert_eq!(faces.count(), 0);
}

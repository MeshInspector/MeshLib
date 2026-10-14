use crate::common::create_cube;
use meshlib::mr::{EdgeId, VertId};

#[test]
fn pack()
{
    let mut mesh = create_cube();
    mesh.topology_mut().pack_4(None, None, None, None);
}

#[test]
fn get_valid_verts()
{
    let mesh = create_cube();
    assert_eq!(mesh.topology().get_valid_verts().count(), 8);
}

#[test]
fn get_valid_faces()
{
    let mesh = create_cube();
    assert_eq!(mesh.topology().get_valid_faces().count(), 12);
}

#[test]
fn find_hole_representive_edges()
{
    let mesh = create_cube();
    let holes = mesh.topology().find_hole_representive_edges(None);
    assert_eq!(holes.size(), 0);
}

#[test]
fn get_left_tri_verts_4()
{
    let mesh = create_cube();
    let (mut v0, mut v1, mut v2) = (VertId {id_: -1}, VertId {id_: -1}, VertId {id_: -1});
    mesh.topology().get_left_tri_verts_4(EdgeId {id_: 0}, &mut v0, &mut v1, &mut v2);
    assert!(v0.id_ >= 0 && v1.id_ >= 0 && v2.id_ >= 0);
}

#[test]
fn find_num_holes()
{
    let mesh = create_cube();
    assert_eq!(mesh.topology().find_num_holes(None), 0);
}

#[test]
fn face_size()
{
    let mesh = create_cube();
    assert_eq!(mesh.topology().face_size(), 12);
}

#[test]
fn get_triangulation()
{
    let mesh = create_cube();
    assert_eq!(mesh.topology().get_triangulation().size(), 12);
}

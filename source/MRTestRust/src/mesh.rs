use crate::common::{create_cube, make_torus};
use meshlib::mr;
use meshlib::mr::{FaceId, UndirectedEdgeId, VertId};

#[test]
fn area()
{
    let mut mesh = create_cube();
    let area = mesh.area_const_mr_face_bit_set_ptr(None);
    assert!(area > 5.999 && area < 6.001);

    let mut faces = mr::FaceBitSet::new();
    faces.resize(12, Some(true));
    for i in 6..12
    {
        faces.set_2(FaceId {id_: i}, false);
    }
    let area = mesh.area_const_mr_face_bit_set_ptr(Some(&faces));
    assert!(area > 2.999 && area < 3.001);

    mesh.delete_faces(&faces, None);
    let area = mesh.area_const_mr_face_bit_set_ptr(None);
    assert!(area > 2.999 && area < 3.001);
}

#[test]
fn short_edges()
{
    let mesh = make_torus(1.0, 0.05, 16, 16);
    let mp = mr::MeshPart::construct(&mesh, None);
    let mut short_edges = mr::find_short_edges(&mp, 0.1, None);
    let short_edges = short_edges.value_mut().expect("findShortEdges failed");
    assert_eq!(short_edges.count(), 256);
}

#[test]
fn incident_faces_from_verts()
{
    let mesh = create_cube();
    let mut verts = mr::VertBitSet::new();
    verts.resize(8, Some(false));
    verts.set_2(VertId {id_: 0}, true);
    let faces = mr::get_incident_faces_mr_vert_bit_set(mesh.topology(), &verts);
    assert_eq!(faces.count(), 6);
}

#[test]
fn incident_faces_from_edges()
{
    let mesh = create_cube();
    let mut edges = mr::UndirectedEdgeBitSet::new();
    edges.resize(12, Some(false));
    edges.set_2(UndirectedEdgeId {id_: 0}, true);
    let faces = mr::get_incident_faces_mr_undirected_edge_bit_set(mesh.topology(), &edges);
    assert_eq!(faces.count(), 8);
}

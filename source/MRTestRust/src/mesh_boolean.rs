use crate::common::make_torus;
use meshlib::mr;
use meshlib::mr::{AffineXf3f, BooleanOperation, BooleanResultMapper_MapObject as MapObject, FaceId, Matrix3f, Turn, Vector3f, VertId};

const PI_F: f32 = 3.14159265358979;

#[test]
fn mesh_boolean()
{
    let mesh_a = make_torus(1.1, 0.5, 8, 8);
    let mut mesh_b = make_torus(1.0, 0.2, 8, 8);
    {
        let rot = Matrix3f::rotation_mr_vector3f(&Vector3f::plus_z(), &Vector3f::plus_y());
        mesh_b.transform(&AffineXf3f::linear(&rot), None);
    }

    let shift_step = 0.2f32;
    let angle_step = PI_F;
    let base_axis = [Vector3f::plus_x(), Vector3f::plus_y(), Vector3f::plus_z()];
    for mask_trans in 0..8
    {
        for mask_rot in 0..8
        {
            let mut shift = 0.01f32;
            while shift < 0.2
            {
                let mut shift_vec = Vector3f {x: 0.0, y: 0.0, z: 0.0};
                for i in 0..3
                {
                    if mask_trans & (1 << i) != 0
                    {
                        shift_vec = mr::op_add_mr_vector3f(&shift_vec, &base_axis[i]);
                    }
                }
                let mut angle = PI_F * 0.01;
                while angle < PI_F * 7.0 / 18.0
                {
                    let mut rotation = Matrix3f::identity();
                    for i in 0..3
                    {
                        if mask_rot & (1 << i) != 0
                        {
                            let rot = Matrix3f::rotation_float(&base_axis[i], angle);
                            rotation = mr::op_mul_mr_matrix3f(&rot, &rotation);
                        }
                    }
                    let xf = mr::op_mul_mr_affine_xf3f(&AffineXf3f::translation(&shift_vec), &AffineXf3f::linear(&rotation));
                    let mut params = mr::BooleanParameters::new();
                    params.set_rigid_b2_a(Some(&xf));
                    let result_ab = mr::boolean_4_const_mr_mesh_ref(&mesh_a, &mesh_b, BooleanOperation::Union, Some(&params));
                    let result_ba = mr::boolean_4_const_mr_mesh_ref(&mesh_b, &mesh_a, BooleanOperation::Union, Some(&params));
                    assert!(result_ab.valid());
                    assert!(result_ba.valid());
                    angle += angle_step;
                }
                shift += shift_step;
            }
        }
    }
}

#[test]
fn boolean_multiple_edge_propogation_sort()
{
    let mut points = mr::VertCoords::construct_1_size_t(6);
    *points.index_mut(VertId {id_: 0}) = Vector3f {x: 0.0, y: 0.0, z: 0.0};
    *points.index_mut(VertId {id_: 1}) = Vector3f {x: -0.5, y: 1.0, z: 0.0};
    *points.index_mut(VertId {id_: 2}) = Vector3f {x: 0.5, y: 1.0, z: 0.0};
    *points.index_mut(VertId {id_: 3}) = Vector3f {x: 0.0, y: 1.5, z: 0.5};
    *points.index_mut(VertId {id_: 4}) = Vector3f {x: -1.0, y: 1.5, z: 0.0};
    *points.index_mut(VertId {id_: 5}) = Vector3f {x: 1.0, y: 1.5, z: 0.0};
    let mut triangulation = mr::Triangulation::construct_1_size_t(5);
    let tri = |a, b, c| meshlib::std_array_MR_VertId_3 {elems: [VertId {id_: a}, VertId {id_: b}, VertId {id_: c}]};
    *triangulation.index_mut(FaceId {id_: 0}) = tri(0, 2, 1);
    *triangulation.index_mut(FaceId {id_: 1}) = tri(1, 2, 3);
    *triangulation.index_mut(FaceId {id_: 2}) = tri(3, 4, 1);
    *triangulation.index_mut(FaceId {id_: 3}) = tri(2, 5, 3);
    *triangulation.index_mut(FaceId {id_: 4}) = tri(3, 5, 4);
    let mut mesh_a = mr::Mesh::from_triangles(&points, &triangulation, None, None);
    {
        let mut mesh_a_sup = mesh_a.clone();
        *mesh_a_sup.points_mut().index_mut(VertId {id_: 3}) = Vector3f {x: 0.0, y: 1.5, z: -0.5};
        let holes = mesh_a.topology().find_hole_representive_edges(None);
        let border = mr::track_right_boundary_loop(mesh_a.topology(), *holes.at(0), None, Some(Turn::Rightmost));
        let mut border_vec = meshlib::std_vector_std_vector_MR_EdgeId::new();
        border_vec.push_back(&border);
        let mp = mr::MeshPart::construct(&mesh_a_sup, Some(mesh_a_sup.topology().get_valid_faces()));
        mesh_a.add_mesh_part_6(&mp, Some(true), Some(&border_vec), Some(&border_vec), None, None);
    }

    let mut mesh_b = mr::make_cube(Some(&Vector3f::diagonal(2.0)), Some(&Vector3f::diagonal(-0.5)));
    mesh_b.transform(&AffineXf3f::translation(&Vector3f {x: -1.5, y: -0.2, z: -0.5}), None);

    for i in 0..BooleanOperation::Count.0
    {
        let result_ab = mr::boolean_4_const_mr_mesh_ref(&mesh_a, &mesh_b, BooleanOperation(i), None);
        let result_ba = mr::boolean_4_const_mr_mesh_ref(&mesh_b, &mesh_a, BooleanOperation(i), None);
        assert!(result_ab.valid());
        assert!(result_ba.valid());
    }
}

#[test]
fn boolean_mapper()
{
    let mesh_a = make_torus(1.1, 0.5, 8, 8);
    let mut mesh_b = make_torus(1.0, 0.2, 8, 8);
    {
        let rot = Matrix3f::rotation_mr_vector3f(&Vector3f::plus_z(), &Vector3f::plus_y());
        mesh_b.transform(&AffineXf3f::linear(&rot), None);
    }

    let mut mapper = mr::BooleanResultMapper::new();
    let result = {
        let mut parameters = mr::BooleanParameters::new();
        parameters.set_mapper(Some(&mut mapper));
        mr::boolean_4_const_mr_mesh_ref(&mesh_a, &mesh_b, BooleanOperation::Union, Some(&parameters))
    };
    assert!(result.valid());

    let v_map_a = mapper.map_mr_vert_bit_set(mesh_a.topology().get_valid_verts(), MapObject::A);
    let v_map_b = mapper.map_mr_vert_bit_set(mesh_b.topology().get_valid_verts(), MapObject::B);
    assert_eq!(mr::op_sub_mr_vert_bit_set(&v_map_a, &v_map_b).count(), v_map_a.count()); // Disjoint.
    assert_eq!(v_map_a.count(), 60);
    assert_eq!(v_map_b.count(), 48);

    let f_map_a = mapper.map_mr_face_bit_set(mesh_a.topology().get_valid_faces(), MapObject::A);
    let f_map_b = mapper.map_mr_face_bit_set(mesh_b.topology().get_valid_faces(), MapObject::B);
    assert_eq!(mr::op_sub_mr_face_bit_set(&f_map_a, &f_map_b).count(), f_map_a.count()); // Disjoint.
    assert_eq!(f_map_a.count(), 224);
    assert_eq!(f_map_b.count(), 192);

    let new_faces = mapper.new_faces();
    assert_eq!(new_faces.size(), 416);
    assert_eq!(new_faces.count(), 252);

    let maps_a = mapper.get_maps(MapObject::A);
    assert_eq!(maps_a.old2new_verts().size(), 160);
    assert_eq!(maps_a.cut2new_faces().size(), 280);
    assert_eq!(maps_a.cut2origin().size(), 280);
    let maps_b = mapper.get_maps(MapObject::B);
    assert_eq!(maps_b.old2new_verts().size(), 160);
    assert_eq!(maps_b.cut2new_faces().size(), 320);
    assert_eq!(maps_b.cut2origin().size(), 320);
}

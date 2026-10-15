use meshlib::mr;
use meshlib::mr::{BooleanOperation, BooleanResultMapper, BooleanResultMapper_MapObject as MapObject, FaceBitSet, FaceId, Vector3f};

fn main() -> Result<(), String>
{
    // First, create a unit sphere.
    let sphere1 = mr::make_uv_sphere(Some(1.0), Some(64), Some(64));

    // Create a copy of this sphere and offset it.
    let mut sphere2 = sphere1.clone();
    let xf = mr::AffineXf3f::translation(&Vector3f {x: 0.7, y: 0.0, z: 0.0});
    sphere2.transform(&xf, None);

    // Ask for an optional mapper relating the primitives of the input meshes to the primitives of the result.
    let mut mapper = BooleanResultMapper::new();

    // Perform the boolean operation.
    let result = mr::boolean_6(&sphere1, &sphere2, BooleanOperation::Intersection, None, Some(&mut mapper), None);
    if !result.valid()
    {
        return Err(format!("Failed to perform boolean: {}", result.error_string()));
    }

    // Find the faces of the result produced by each input sphere, and the faces the cut created.
    let faces_of_sphere1 = mapper.map_mr_face_bit_set(sphere1.topology().get_valid_faces(), MapObject::A);
    let faces_of_sphere2 = mapper.map_mr_face_bit_set(sphere2.topology().get_valid_faces(), MapObject::B);
    let new_faces = mapper.new_faces();

    println!("faces from sphere1: {}", faces_of_sphere1.count());
    println!("faces from sphere2: {}", faces_of_sphere2.count());
    println!("faces created by the cut: {}", new_faces.count());

    // Map one particular face of sphere1 forward: the cut can split it in several faces of the
    // result, or drop it completely if that part of sphere1 is not in the result.
    let face_of_sphere1 = FaceId {id_: 793};
    let mut one_face = FaceBitSet::new();
    one_face.auto_resize_set_2(face_of_sphere1, None);
    let produced_faces = mapper.map_mr_face_bit_set(&one_face, MapObject::A);
    println!("face {} of sphere1 produced {} faces of the result", face_of_sphere1.id_, produced_faces.count());

    // And backward: the face of sphere1 each face of the result came from
    // (invalid id for the faces that came from sphere2).
    let new2old_faces = mapper.get_new2_old_face_map(MapObject::A);
    let result_face = produced_faces.find_first();
    println!("face {} of the result came from face {} of sphere1", result_face.id_, new2old_faces.index(result_face).id_);

    // Save result to an STL file.
    let saved = mr::mesh_save::to_any_supported_format_3(result.mesh(), "out_boolean.stl", None);
    if let Some(error) = saved.error()
    {
        return Err(format!("Failed to save mesh: {error}"));
    }

    Ok(())
}

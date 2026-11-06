use meshlib::{EdgeWeights, Laplacian, VertBitSet, VertexMass};

fn main() -> Result<(), String>
{
    // Load mesh
    let mut mesh = meshlib::mesh_load::from_any_supported_format("mesh.stl", None).into_result()?;

    // Construct deformer on the mesh vertices
    let mut l_deformer = Laplacian::construct(&mut mesh);

    // Find an area for the deformation anchor points
    let anc_v0 = mesh.topology().get_valid_verts().find_first();
    let anc_v1 = mesh.topology().get_valid_verts().find_last();
    // Mark the anchor points in the free area
    let mut free_verts = VertBitSet::new();
    free_verts.resize(mesh.topology().get_valid_verts().size(), None);
    free_verts.set(anc_v0, true);
    free_verts.set(anc_v1, true);
    // Expand the free area
    meshlib::expand_mr_vert_bit_set(mesh.topology(), &mut free_verts, Some(5));

    // Initialize laplacian
    l_deformer.init(&free_verts, EdgeWeights::Cotan, Some(VertexMass::NeiArea), None);

    let shift_amount = mesh.compute_bounding_box(None).diagonal() * 0.01;
    // Fix the anchor vertices in the required position
    l_deformer.fix_vertex_pos(anc_v0, &meshlib::op_add_mr_vector3f(mesh.points().index(anc_v0), &meshlib::op_mul_mr_vector3f_float(&mesh.normal_mr_vert_id(anc_v0), shift_amount)), None);
    l_deformer.fix_vertex_pos(anc_v1, &meshlib::op_add_mr_vector3f(mesh.points().index(anc_v1), &meshlib::op_mul_mr_vector3f_float(&mesh.normal_mr_vert_id(anc_v1), shift_amount)), None);

    // Move the free vertices according to the anchor ones
    l_deformer.apply();

    // Invalidate the mesh because of the external vertex changes
    mesh.invalidate_caches(None);

    // Save the deformed mesh
    meshlib::mesh_save::to_any_supported_format(&mesh, "deformed_mesh.stl", None).into_result()?;

    Ok(())
}

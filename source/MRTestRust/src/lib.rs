//! Unit tests of the generated Rust bindings, ported from `MRTestC2`.

#![cfg(test)]
#![allow(non_snake_case)]

mod common;
mod add_noise;
mod affine_xf;
mod bit_set;
mod r#box;
mod color;
mod expand_shrink;
mod fix_self_intersections;
mod matrix3;
mod mesh;
mod mesh_boolean;
mod mesh_builder;
mod mesh_collide;
mod mesh_collide_precise;
mod mesh_components;
mod mesh_decimate;
mod mesh_extrude;
mod mesh_fill_hole;
mod mesh_mesh_distance;
mod mesh_normals;
mod mesh_offset;
mod mesh_subdivide;
mod mesh_topology;
mod point_cloud;
mod points_to_mesh_projector;
mod simple_objects;
mod vdb_conversions;
mod vector3;

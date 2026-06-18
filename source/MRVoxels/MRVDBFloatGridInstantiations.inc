// Explicit-instantiation list for openvdb::FloatGrid and its tree (FloatTree =
// tree::Tree4<float,5,4,3>::Type). Included from MRVoxelsPch.h with MR_VDB_INSTANTIATE =
// `extern template` (so each voxels TU skips this codegen) and from
// MRVDBFloatGridInstantiation.cpp with `template` (the single definition they link against).
// Only Grid and Tree are instantiated -- they transitively home the heavy RootNode/InternalNode/
// LeafNode methods reached through them; the raw nodes cannot be explicitly instantiated
// (LeafNode::str() references a non-existent operator<< for LeafBuffer).
MR_VDB_INSTANTIATE class openvdb::tree::Tree<openvdb::tree::RootNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::LeafNode<float, 3>, 4>, 5>>>;
MR_VDB_INSTANTIATE class openvdb::Grid<openvdb::FloatTree>;

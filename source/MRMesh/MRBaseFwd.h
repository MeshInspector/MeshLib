#pragma once

#include "exports.h"

#include "config.h"

#include "MRKernelFwd.h"

#include <parallel_hashmap/phmap_fwd_decl.h>

namespace MR
{

class PointToPointAligningTransform;
class PointToPlaneAligningTransform;

template <typename T, typename Hash = phmap::priv::hash_default_hash<T>, typename Eq = phmap::priv::hash_default_eq<T>>
using HashSet = phmap::flat_hash_set<T, Hash, Eq>;
template <typename T, typename Hash = phmap::priv::hash_default_hash<T>, typename Eq = phmap::priv::hash_default_eq<T>>
using ParallelHashSet = phmap::parallel_flat_hash_set<T, Hash, Eq>;

template <typename K, typename V, typename Hash = phmap::priv::hash_default_hash<K>, typename Eq = phmap::priv::hash_default_eq<K>>
using HashMap = phmap::flat_hash_map<K, V, Hash, Eq>;
template <typename K, typename V, typename Hash = phmap::priv::hash_default_hash<K>, typename Eq = phmap::priv::hash_default_eq<K>>
using ParallelHashMap = phmap::parallel_flat_hash_map<K, V, Hash, Eq>;

class MRMESH_CLASS BitSet;
template <typename T> class MRMESH_CLASS TaggedBitSet;
using VertBitSet = TaggedBitSet<VertTag>;
using EdgeBitSet = TaggedBitSet<EdgeTag>;
using UndirectedEdgeBitSet = TaggedBitSet<UndirectedEdgeTag>;
using FaceBitSet = TaggedBitSet<FaceTag>;
using PixelBitSet = TaggedBitSet<PixelTag>;
using VoxelBitSet = TaggedBitSet<VoxelTag>;

template <typename T> class MRMESH_CLASS SetBitIteratorT;
using SetBitIterator = SetBitIteratorT<BitSet>;
using VertSetBitIterator = SetBitIteratorT<VertBitSet>;
using EdgeSetBitIterator = SetBitIteratorT<EdgeBitSet>;
using UndirectedEdgeSetBitIterator = SetBitIteratorT<UndirectedEdgeBitSet>;
using FaceSetBitIterator = SetBitIteratorT<FaceBitSet>;

template <typename T, typename I, typename P> class Heap;

template <typename Tag>
class MRMESH_CLASS ColorMapAggregator;

class Graph;
class GraphVertTag;
class GraphEdgeTag;
using GraphVertId = Id<GraphVertTag>;
using GraphEdgeId = Id<GraphEdgeTag>;
using GraphVertBitSet = TaggedBitSet<GraphVertTag>;
using GraphEdgeBitSet = TaggedBitSet<GraphEdgeTag>;

} // namespace MR

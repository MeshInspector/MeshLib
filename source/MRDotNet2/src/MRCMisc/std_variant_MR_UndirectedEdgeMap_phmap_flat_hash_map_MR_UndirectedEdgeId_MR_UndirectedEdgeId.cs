public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 2 objects: `MR::UndirectedEdgeMap`, `phmap::flat_hash_map<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
        /// This is the const half of the class.
        public class Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy(_Underlying *_this);
                __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR.Std._ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Index(_Underlying *_this);
                return __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `MR::UndirectedEdgeMap`.
            public unsafe Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR._ByValue_UndirectedEdgeMap value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_MR_UndirectedEdgeMap", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_MR_UndirectedEdgeMap(MR.Misc._PassBy value_pass_by, MR.UndirectedEdgeMap._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_MR_UndirectedEdgeMap(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `phmap::flat_hash_map<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
            public unsafe Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR.Phmap._ByValue_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::UndirectedEdgeMap`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_UndirectedEdgeMap? GetMRUndirectedEdgeMap()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_MR_UndirectedEdgeMap", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeMap._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_MR_UndirectedEdgeMap(_Underlying *_this);
                var __ret = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_MR_UndirectedEdgeMap(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_UndirectedEdgeMap(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `phmap::flat_hash_map<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? GetPhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
                extern static MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_Underlying *_this);
                var __ret = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_UnderlyingPtr);
                return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 2 objects: `MR::UndirectedEdgeMap`, `phmap::flat_hash_map<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
        /// This is the non-const half of the class.
        public class Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId : Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId
        {
            internal unsafe Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR.Std._ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *other);
                __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 0, of type `MR::UndirectedEdgeMap`.
            public unsafe Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR._ByValue_UndirectedEdgeMap value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_MR_UndirectedEdgeMap", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_MR_UndirectedEdgeMap(MR.Misc._PassBy value_pass_by, MR.UndirectedEdgeMap._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_MR_UndirectedEdgeMap(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `phmap::flat_hash_map<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
            public unsafe Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR.Phmap._ByValue_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 0, of type `MR::UndirectedEdgeMap`.
            public unsafe void AssignAsMRUndirectedEdgeMap(MR._ByValue_UndirectedEdgeMap value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignAs_MR_UndirectedEdgeMap", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignAs_MR_UndirectedEdgeMap(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.UndirectedEdgeMap._Underlying *value);
                __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignAs_MR_UndirectedEdgeMap(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 1, of type `phmap::flat_hash_map<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
            public unsafe void AssignAsPhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR.Phmap._ByValue_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *value);
                __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignAs_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::UndirectedEdgeMap`, mutable. If it's not the active element, returns null.
            public unsafe MR.UndirectedEdgeMap? GetMutableMRUndirectedEdgeMap()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_MR_UndirectedEdgeMap", ExactSpelling = true)]
                extern static MR.UndirectedEdgeMap._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_MR_UndirectedEdgeMap(_Underlying *_this);
                var __ret = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_MR_UndirectedEdgeMap(_UnderlyingPtr);
                return __ret is not null ? new MR.UndirectedEdgeMap(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `phmap::flat_hash_map<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? GetMutablePhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_Underlying *_this);
                var __ret = __MR_std_variant_MR_UndirectedEdgeMap_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_UnderlyingPtr);
                return __ret is not null ? new MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId`/`Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId
        {
            internal readonly Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId arg) {return new(arg);}
            public _ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR.Misc._Moved<Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(MR.Misc._Moved<Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId`/`Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId` directly.
        public class _InOptMut_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId
        {
            public Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId? Opt;

            public _InOptMut_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId() {}
            public _InOptMut_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId`/`Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId` to pass it to the function.
        public class _InOptConst_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId
        {
            public Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId? Opt;

            public _InOptConst_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId() {}
            public _InOptConst_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId(Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId value) {return new(value);}
        }
    }
}

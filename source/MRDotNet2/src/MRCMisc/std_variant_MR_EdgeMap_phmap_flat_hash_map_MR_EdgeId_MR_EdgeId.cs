public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 2 objects: `MR::EdgeMap`, `phmap::flat_hash_map<MR::EdgeId, MR::EdgeId>`.
        /// This is the const half of the class.
        public class Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Destroy(_Underlying *_this);
                __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(MR.Std._ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Index(_Underlying *_this);
                return __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `MR::EdgeMap`.
            public unsafe Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(MR._ByValue_EdgeMap value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_MR_EdgeMap", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_MR_EdgeMap(MR.Misc._PassBy value_pass_by, MR.EdgeMap._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_MR_EdgeMap(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `phmap::flat_hash_map<MR::EdgeId, MR::EdgeId>`.
            public unsafe Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(MR.Phmap._ByValue_FlatHashMap_MREdgeId_MREdgeId value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MREdgeId_MREdgeId._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::EdgeMap`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_EdgeMap? GetMREdgeMap()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Get_MR_EdgeMap", ExactSpelling = true)]
                extern static MR.Const_EdgeMap._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Get_MR_EdgeMap(_Underlying *_this);
                var __ret = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Get_MR_EdgeMap(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_EdgeMap(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `phmap::flat_hash_map<MR::EdgeId, MR::EdgeId>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId? GetPhmapFlatHashMapMREdgeIdMREdgeId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Get_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
                extern static MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Get_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_Underlying *_this);
                var __ret = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_Get_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_UnderlyingPtr);
                return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 2 objects: `MR::EdgeMap`, `phmap::flat_hash_map<MR::EdgeId, MR::EdgeId>`.
        /// This is the non-const half of the class.
        public class Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId : Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId
        {
            internal unsafe Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(MR.Std._ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *other);
                __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 0, of type `MR::EdgeMap`.
            public unsafe Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(MR._ByValue_EdgeMap value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_MR_EdgeMap", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_MR_EdgeMap(MR.Misc._PassBy value_pass_by, MR.EdgeMap._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_MR_EdgeMap(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `phmap::flat_hash_map<MR::EdgeId, MR::EdgeId>`.
            public unsafe Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(MR.Phmap._ByValue_FlatHashMap_MREdgeId_MREdgeId value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MREdgeId_MREdgeId._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_ConstructAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 0, of type `MR::EdgeMap`.
            public unsafe void AssignAsMREdgeMap(MR._ByValue_EdgeMap value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignAs_MR_EdgeMap", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignAs_MR_EdgeMap(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.EdgeMap._Underlying *value);
                __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignAs_MR_EdgeMap(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 1, of type `phmap::flat_hash_map<MR::EdgeId, MR::EdgeId>`.
            public unsafe void AssignAsPhmapFlatHashMapMREdgeIdMREdgeId(MR.Phmap._ByValue_FlatHashMap_MREdgeId_MREdgeId value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MREdgeId_MREdgeId._Underlying *value);
                __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_AssignAs_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::EdgeMap`, mutable. If it's not the active element, returns null.
            public unsafe MR.EdgeMap? GetMutableMREdgeMap()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_GetMutable_MR_EdgeMap", ExactSpelling = true)]
                extern static MR.EdgeMap._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_GetMutable_MR_EdgeMap(_Underlying *_this);
                var __ret = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_GetMutable_MR_EdgeMap(_UnderlyingPtr);
                return __ret is not null ? new MR.EdgeMap(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `phmap::flat_hash_map<MR::EdgeId, MR::EdgeId>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Phmap.FlatHashMap_MREdgeId_MREdgeId? GetMutablePhmapFlatHashMapMREdgeIdMREdgeId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_GetMutable_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MREdgeId_MREdgeId._Underlying *__MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_GetMutable_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_Underlying *_this);
                var __ret = __MR_std_variant_MR_EdgeMap_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId_GetMutable_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_UnderlyingPtr);
                return __ret is not null ? new MR.Phmap.FlatHashMap_MREdgeId_MREdgeId(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId`/`Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId
        {
            internal readonly Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId arg) {return new(arg);}
            public _ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(MR.Misc._Moved<Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(MR.Misc._Moved<Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId`/`Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId` directly.
        public class _InOptMut_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId
        {
            public Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId? Opt;

            public _InOptMut_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId() {}
            public _InOptMut_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId`/`Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId` to pass it to the function.
        public class _InOptConst_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId
        {
            public Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId? Opt;

            public _InOptConst_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId() {}
            public _InOptConst_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId(Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId value) {return new(value);}
        }
    }
}

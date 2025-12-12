public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 2 objects: `MR::VertMap`, `phmap::flat_hash_map<MR::VertId, MR::VertId>`.
        /// This is the const half of the class.
        public class Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Destroy(_Underlying *_this);
                __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(MR.Std._ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Index(_Underlying *_this);
                return __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `MR::VertMap`.
            public unsafe Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(MR._ByValue_VertMap value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_MR_VertMap", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_MR_VertMap(MR.Misc._PassBy value_pass_by, MR.VertMap._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_MR_VertMap(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `phmap::flat_hash_map<MR::VertId, MR::VertId>`.
            public unsafe Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(MR.Phmap._ByValue_FlatHashMap_MRVertId_MRVertId value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_phmap_flat_hash_map_MR_VertId_MR_VertId", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_phmap_flat_hash_map_MR_VertId_MR_VertId(MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_phmap_flat_hash_map_MR_VertId_MR_VertId(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::VertMap`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_VertMap? GetMRVertMap()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Get_MR_VertMap", ExactSpelling = true)]
                extern static MR.Const_VertMap._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Get_MR_VertMap(_Underlying *_this);
                var __ret = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Get_MR_VertMap(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_VertMap(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `phmap::flat_hash_map<MR::VertId, MR::VertId>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId? GetPhmapFlatHashMapMRVertIdMRVertId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Get_phmap_flat_hash_map_MR_VertId_MR_VertId", ExactSpelling = true)]
                extern static MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Get_phmap_flat_hash_map_MR_VertId_MR_VertId(_Underlying *_this);
                var __ret = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_Get_phmap_flat_hash_map_MR_VertId_MR_VertId(_UnderlyingPtr);
                return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 2 objects: `MR::VertMap`, `phmap::flat_hash_map<MR::VertId, MR::VertId>`.
        /// This is the non-const half of the class.
        public class Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId : Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId
        {
            internal unsafe Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(MR.Std._ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *other);
                __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 0, of type `MR::VertMap`.
            public unsafe Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(MR._ByValue_VertMap value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_MR_VertMap", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_MR_VertMap(MR.Misc._PassBy value_pass_by, MR.VertMap._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_MR_VertMap(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `phmap::flat_hash_map<MR::VertId, MR::VertId>`.
            public unsafe Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(MR.Phmap._ByValue_FlatHashMap_MRVertId_MRVertId value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_phmap_flat_hash_map_MR_VertId_MR_VertId", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_phmap_flat_hash_map_MR_VertId_MR_VertId(MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_ConstructAs_phmap_flat_hash_map_MR_VertId_MR_VertId(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 0, of type `MR::VertMap`.
            public unsafe void AssignAsMRVertMap(MR._ByValue_VertMap value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignAs_MR_VertMap", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignAs_MR_VertMap(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.VertMap._Underlying *value);
                __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignAs_MR_VertMap(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 1, of type `phmap::flat_hash_map<MR::VertId, MR::VertId>`.
            public unsafe void AssignAsPhmapFlatHashMapMRVertIdMRVertId(MR.Phmap._ByValue_FlatHashMap_MRVertId_MRVertId value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignAs_phmap_flat_hash_map_MR_VertId_MR_VertId", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignAs_phmap_flat_hash_map_MR_VertId_MR_VertId(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *value);
                __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_AssignAs_phmap_flat_hash_map_MR_VertId_MR_VertId(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::VertMap`, mutable. If it's not the active element, returns null.
            public unsafe MR.VertMap? GetMutableMRVertMap()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_GetMutable_MR_VertMap", ExactSpelling = true)]
                extern static MR.VertMap._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_GetMutable_MR_VertMap(_Underlying *_this);
                var __ret = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_GetMutable_MR_VertMap(_UnderlyingPtr);
                return __ret is not null ? new MR.VertMap(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `phmap::flat_hash_map<MR::VertId, MR::VertId>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Phmap.FlatHashMap_MRVertId_MRVertId? GetMutablePhmapFlatHashMapMRVertIdMRVertId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_GetMutable_phmap_flat_hash_map_MR_VertId_MR_VertId", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *__MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_GetMutable_phmap_flat_hash_map_MR_VertId_MR_VertId(_Underlying *_this);
                var __ret = __MR_std_variant_MR_VertMap_phmap_flat_hash_map_MR_VertId_MR_VertId_GetMutable_phmap_flat_hash_map_MR_VertId_MR_VertId(_UnderlyingPtr);
                return __ret is not null ? new MR.Phmap.FlatHashMap_MRVertId_MRVertId(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId`/`Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId
        {
            internal readonly Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId arg) {return new(arg);}
            public _ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(MR.Misc._Moved<Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(MR.Misc._Moved<Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId`/`Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId` directly.
        public class _InOptMut_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId
        {
            public Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId? Opt;

            public _InOptMut_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId() {}
            public _InOptMut_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId`/`Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId` to pass it to the function.
        public class _InOptConst_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId
        {
            public Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId? Opt;

            public _InOptConst_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId() {}
            public _InOptConst_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId(Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId value) {return new(value);}
        }
    }
}

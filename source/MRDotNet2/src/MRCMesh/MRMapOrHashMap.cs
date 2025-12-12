public static partial class MR
{
    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::FaceId, MR::FaceId>`.
    /// This is the const half of the class.
    public class Const_MapOrHashMap_MRFaceId_MRFaceId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MapOrHashMap_MRFaceId_MRFaceId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_Destroy", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_FaceId_MR_FaceId_Destroy(_Underlying *_this);
            __MR_MapOrHashMap_MR_FaceId_MR_FaceId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MapOrHashMap_MRFaceId_MRFaceId() {Dispose(false);}

        // default construction will select dense map
        public unsafe MR.Std.Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_Get_var", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_Get_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_FaceId_MR_FaceId_Get_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MapOrHashMap_MRFaceId_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::FaceId, MR::FaceId>` elementwise.
        public unsafe Const_MapOrHashMap_MRFaceId_MRFaceId(MR.Std._ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::MapOrHashMap`.
        public unsafe Const_MapOrHashMap_MRFaceId_MRFaceId(MR._ByValue_MapOrHashMap_MRFaceId_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::createMap`.
        /// Parameter `size` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MRFaceId_MRFaceId> CreateMap(ulong? size = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_createMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_createMap(ulong *size);
            ulong __deref_size = size.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MRFaceId_MRFaceId(__MR_MapOrHashMap_MR_FaceId_MR_FaceId_createMap(size.HasValue ? &__deref_size : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::createHashMap`.
        /// Parameter `capacity` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MRFaceId_MRFaceId> CreateHashMap(ulong? capacity = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_createHashMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_createHashMap(ulong *capacity);
            ulong __deref_capacity = capacity.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MRFaceId_MRFaceId(__MR_MapOrHashMap_MR_FaceId_MR_FaceId_createHashMap(capacity.HasValue ? &__deref_capacity : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::getMap`.
        public unsafe MR.Const_FaceMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_getMap_const", ExactSpelling = true)]
            extern static MR.Const_FaceMap._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_getMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_getMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_FaceMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::getHashMap`.
        public unsafe MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_getHashMap_const", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_getHashMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_getHashMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId(__ret, is_owning: false) : null;
        }
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::FaceId, MR::FaceId>`.
    /// This is the non-const half of the class.
    public class MapOrHashMap_MRFaceId_MRFaceId : Const_MapOrHashMap_MRFaceId_MRFaceId
    {
        internal unsafe MapOrHashMap_MRFaceId_MRFaceId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // default construction will select dense map
        public new unsafe MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_GetMutable_var", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_GetMutable_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_FaceId_MR_FaceId_GetMutable_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MapOrHashMap_MRFaceId_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::FaceId, MR::FaceId>` elementwise.
        public unsafe MapOrHashMap_MRFaceId_MRFaceId(MR.Std._ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::MapOrHashMap`.
        public unsafe MapOrHashMap_MRFaceId_MRFaceId(MR._ByValue_MapOrHashMap_MRFaceId_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::operator=`.
        public unsafe MR.MapOrHashMap_MRFaceId_MRFaceId Assign(MR._ByValue_MapOrHashMap_MRFaceId_MRFaceId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *_other);
            return new(__MR_MapOrHashMap_MR_FaceId_MR_FaceId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::setMap`.
        public unsafe void SetMap(MR.Misc._Moved<MR.FaceMap> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_setMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_FaceId_MR_FaceId_setMap(_Underlying *_this, MR.FaceMap._Underlying *m);
            __MR_MapOrHashMap_MR_FaceId_MR_FaceId_setMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::setHashMap`.
        public unsafe void SetHashMap(MR.Misc._Moved<MR.Phmap.FlatHashMap_MRFaceId_MRFaceId> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_setHashMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_FaceId_MR_FaceId_setHashMap(_Underlying *_this, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *m);
            __MR_MapOrHashMap_MR_FaceId_MR_FaceId_setHashMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// if this stores dense map then resizes it to denseTotalSize;
        /// if this stores hash map then sets its capacity to size()+hashAdditionalCapacity
        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::resizeReserve`.
        public unsafe void ResizeReserve(ulong denseTotalSize, ulong hashAdditionalCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_resizeReserve", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_FaceId_MR_FaceId_resizeReserve(_Underlying *_this, ulong denseTotalSize, ulong hashAdditionalCapacity);
            __MR_MapOrHashMap_MR_FaceId_MR_FaceId_resizeReserve(_UnderlyingPtr, denseTotalSize, hashAdditionalCapacity);
        }

        /// appends one element in the map,
        /// in case of dense map, key must be equal to vector.endId()
        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::pushBack`.
        public unsafe void PushBack(MR.FaceId key, MR.FaceId val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_pushBack", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_FaceId_MR_FaceId_pushBack(_Underlying *_this, MR.FaceId key, MR.FaceId val);
            __MR_MapOrHashMap_MR_FaceId_MR_FaceId_pushBack(_UnderlyingPtr, key, val);
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::getMap`.
        public unsafe new MR.FaceMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_getMap", ExactSpelling = true)]
            extern static MR.FaceMap._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_getMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_getMap(_UnderlyingPtr);
            return __ret is not null ? new MR.FaceMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::getHashMap`.
        public unsafe new MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_getHashMap", ExactSpelling = true)]
            extern static MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *__MR_MapOrHashMap_MR_FaceId_MR_FaceId_getHashMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_FaceId_MR_FaceId_getHashMap(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.FlatHashMap_MRFaceId_MRFaceId(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::FaceId, MR::FaceId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_FaceId_MR_FaceId_clear", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_FaceId_MR_FaceId_clear(_Underlying *_this);
            __MR_MapOrHashMap_MR_FaceId_MR_FaceId_clear(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MapOrHashMap_MRFaceId_MRFaceId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MapOrHashMap_MRFaceId_MRFaceId`/`Const_MapOrHashMap_MRFaceId_MRFaceId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MapOrHashMap_MRFaceId_MRFaceId
    {
        internal readonly Const_MapOrHashMap_MRFaceId_MRFaceId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MapOrHashMap_MRFaceId_MRFaceId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MapOrHashMap_MRFaceId_MRFaceId(Const_MapOrHashMap_MRFaceId_MRFaceId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MapOrHashMap_MRFaceId_MRFaceId(Const_MapOrHashMap_MRFaceId_MRFaceId arg) {return new(arg);}
        public _ByValue_MapOrHashMap_MRFaceId_MRFaceId(MR.Misc._Moved<MapOrHashMap_MRFaceId_MRFaceId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MapOrHashMap_MRFaceId_MRFaceId(MR.Misc._Moved<MapOrHashMap_MRFaceId_MRFaceId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MRFaceId_MRFaceId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MapOrHashMap_MRFaceId_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MRFaceId_MRFaceId`/`Const_MapOrHashMap_MRFaceId_MRFaceId` directly.
    public class _InOptMut_MapOrHashMap_MRFaceId_MRFaceId
    {
        public MapOrHashMap_MRFaceId_MRFaceId? Opt;

        public _InOptMut_MapOrHashMap_MRFaceId_MRFaceId() {}
        public _InOptMut_MapOrHashMap_MRFaceId_MRFaceId(MapOrHashMap_MRFaceId_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptMut_MapOrHashMap_MRFaceId_MRFaceId(MapOrHashMap_MRFaceId_MRFaceId value) {return new(value);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MRFaceId_MRFaceId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MapOrHashMap_MRFaceId_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MRFaceId_MRFaceId`/`Const_MapOrHashMap_MRFaceId_MRFaceId` to pass it to the function.
    public class _InOptConst_MapOrHashMap_MRFaceId_MRFaceId
    {
        public Const_MapOrHashMap_MRFaceId_MRFaceId? Opt;

        public _InOptConst_MapOrHashMap_MRFaceId_MRFaceId() {}
        public _InOptConst_MapOrHashMap_MRFaceId_MRFaceId(Const_MapOrHashMap_MRFaceId_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptConst_MapOrHashMap_MRFaceId_MRFaceId(Const_MapOrHashMap_MRFaceId_MRFaceId value) {return new(value);}
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::VertId, MR::VertId>`.
    /// This is the const half of the class.
    public class Const_MapOrHashMap_MRVertId_MRVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MapOrHashMap_MRVertId_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_Destroy", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_VertId_MR_VertId_Destroy(_Underlying *_this);
            __MR_MapOrHashMap_MR_VertId_MR_VertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MapOrHashMap_MRVertId_MRVertId() {Dispose(false);}

        // default construction will select dense map
        public unsafe MR.Std.Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_Get_var", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_Get_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_VertId_MR_VertId_Get_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MapOrHashMap_MRVertId_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_VertId_MR_VertId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::VertId, MR::VertId>` elementwise.
        public unsafe Const_MapOrHashMap_MRVertId_MRVertId(MR.Std._ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::VertId, MR::VertId>::MapOrHashMap`.
        public unsafe Const_MapOrHashMap_MRVertId_MRVertId(MR._ByValue_MapOrHashMap_MRVertId_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRVertId_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::createMap`.
        /// Parameter `size` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MRVertId_MRVertId> CreateMap(ulong? size = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_createMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_createMap(ulong *size);
            ulong __deref_size = size.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MRVertId_MRVertId(__MR_MapOrHashMap_MR_VertId_MR_VertId_createMap(size.HasValue ? &__deref_size : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::createHashMap`.
        /// Parameter `capacity` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MRVertId_MRVertId> CreateHashMap(ulong? capacity = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_createHashMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_createHashMap(ulong *capacity);
            ulong __deref_capacity = capacity.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MRVertId_MRVertId(__MR_MapOrHashMap_MR_VertId_MR_VertId_createHashMap(capacity.HasValue ? &__deref_capacity : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::getMap`.
        public unsafe MR.Const_VertMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_getMap_const", ExactSpelling = true)]
            extern static MR.Const_VertMap._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_getMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_VertId_MR_VertId_getMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_VertMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::getHashMap`.
        public unsafe MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_getHashMap_const", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_getHashMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_VertId_MR_VertId_getHashMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId(__ret, is_owning: false) : null;
        }
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::VertId, MR::VertId>`.
    /// This is the non-const half of the class.
    public class MapOrHashMap_MRVertId_MRVertId : Const_MapOrHashMap_MRVertId_MRVertId
    {
        internal unsafe MapOrHashMap_MRVertId_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // default construction will select dense map
        public new unsafe MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_GetMutable_var", ExactSpelling = true)]
                extern static MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_GetMutable_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_VertId_MR_VertId_GetMutable_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MapOrHashMap_MRVertId_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_VertId_MR_VertId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::VertId, MR::VertId>` elementwise.
        public unsafe MapOrHashMap_MRVertId_MRVertId(MR.Std._ByValue_Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MRVertMap_PhmapFlatHashMapMRVertIdMRVertId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::VertId, MR::VertId>::MapOrHashMap`.
        public unsafe MapOrHashMap_MRVertId_MRVertId(MR._ByValue_MapOrHashMap_MRVertId_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRVertId_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_VertId_MR_VertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::operator=`.
        public unsafe MR.MapOrHashMap_MRVertId_MRVertId Assign(MR._ByValue_MapOrHashMap_MRVertId_MRVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRVertId_MRVertId._Underlying *_other);
            return new(__MR_MapOrHashMap_MR_VertId_MR_VertId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::setMap`.
        public unsafe void SetMap(MR.Misc._Moved<MR.VertMap> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_setMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_VertId_MR_VertId_setMap(_Underlying *_this, MR.VertMap._Underlying *m);
            __MR_MapOrHashMap_MR_VertId_MR_VertId_setMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::setHashMap`.
        public unsafe void SetHashMap(MR.Misc._Moved<MR.Phmap.FlatHashMap_MRVertId_MRVertId> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_setHashMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_VertId_MR_VertId_setHashMap(_Underlying *_this, MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *m);
            __MR_MapOrHashMap_MR_VertId_MR_VertId_setHashMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// if this stores dense map then resizes it to denseTotalSize;
        /// if this stores hash map then sets its capacity to size()+hashAdditionalCapacity
        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::resizeReserve`.
        public unsafe void ResizeReserve(ulong denseTotalSize, ulong hashAdditionalCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_resizeReserve", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_VertId_MR_VertId_resizeReserve(_Underlying *_this, ulong denseTotalSize, ulong hashAdditionalCapacity);
            __MR_MapOrHashMap_MR_VertId_MR_VertId_resizeReserve(_UnderlyingPtr, denseTotalSize, hashAdditionalCapacity);
        }

        /// appends one element in the map,
        /// in case of dense map, key must be equal to vector.endId()
        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::pushBack`.
        public unsafe void PushBack(MR.VertId key, MR.VertId val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_pushBack", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_VertId_MR_VertId_pushBack(_Underlying *_this, MR.VertId key, MR.VertId val);
            __MR_MapOrHashMap_MR_VertId_MR_VertId_pushBack(_UnderlyingPtr, key, val);
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::getMap`.
        public unsafe new MR.VertMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_getMap", ExactSpelling = true)]
            extern static MR.VertMap._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_getMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_VertId_MR_VertId_getMap(_UnderlyingPtr);
            return __ret is not null ? new MR.VertMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::getHashMap`.
        public unsafe new MR.Phmap.FlatHashMap_MRVertId_MRVertId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_getHashMap", ExactSpelling = true)]
            extern static MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *__MR_MapOrHashMap_MR_VertId_MR_VertId_getHashMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_VertId_MR_VertId_getHashMap(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.FlatHashMap_MRVertId_MRVertId(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::VertId, MR::VertId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_VertId_MR_VertId_clear", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_VertId_MR_VertId_clear(_Underlying *_this);
            __MR_MapOrHashMap_MR_VertId_MR_VertId_clear(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MapOrHashMap_MRVertId_MRVertId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MapOrHashMap_MRVertId_MRVertId`/`Const_MapOrHashMap_MRVertId_MRVertId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MapOrHashMap_MRVertId_MRVertId
    {
        internal readonly Const_MapOrHashMap_MRVertId_MRVertId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MapOrHashMap_MRVertId_MRVertId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MapOrHashMap_MRVertId_MRVertId(Const_MapOrHashMap_MRVertId_MRVertId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MapOrHashMap_MRVertId_MRVertId(Const_MapOrHashMap_MRVertId_MRVertId arg) {return new(arg);}
        public _ByValue_MapOrHashMap_MRVertId_MRVertId(MR.Misc._Moved<MapOrHashMap_MRVertId_MRVertId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MapOrHashMap_MRVertId_MRVertId(MR.Misc._Moved<MapOrHashMap_MRVertId_MRVertId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MRVertId_MRVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MapOrHashMap_MRVertId_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MRVertId_MRVertId`/`Const_MapOrHashMap_MRVertId_MRVertId` directly.
    public class _InOptMut_MapOrHashMap_MRVertId_MRVertId
    {
        public MapOrHashMap_MRVertId_MRVertId? Opt;

        public _InOptMut_MapOrHashMap_MRVertId_MRVertId() {}
        public _InOptMut_MapOrHashMap_MRVertId_MRVertId(MapOrHashMap_MRVertId_MRVertId value) {Opt = value;}
        public static implicit operator _InOptMut_MapOrHashMap_MRVertId_MRVertId(MapOrHashMap_MRVertId_MRVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MRVertId_MRVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MapOrHashMap_MRVertId_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MRVertId_MRVertId`/`Const_MapOrHashMap_MRVertId_MRVertId` to pass it to the function.
    public class _InOptConst_MapOrHashMap_MRVertId_MRVertId
    {
        public Const_MapOrHashMap_MRVertId_MRVertId? Opt;

        public _InOptConst_MapOrHashMap_MRVertId_MRVertId() {}
        public _InOptConst_MapOrHashMap_MRVertId_MRVertId(Const_MapOrHashMap_MRVertId_MRVertId value) {Opt = value;}
        public static implicit operator _InOptConst_MapOrHashMap_MRVertId_MRVertId(Const_MapOrHashMap_MRVertId_MRVertId value) {return new(value);}
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>`.
    /// This is the const half of the class.
    public class Const_MapOrHashMap_MREdgeId_MREdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MapOrHashMap_MREdgeId_MREdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_Destroy(_Underlying *_this);
            __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MapOrHashMap_MREdgeId_MREdgeId() {Dispose(false);}

        // default construction will select dense map
        public unsafe MR.Std.Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_Get_var", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_Get_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_Get_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MapOrHashMap_MREdgeId_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>` elementwise.
        public unsafe Const_MapOrHashMap_MREdgeId_MREdgeId(MR.Std._ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::MapOrHashMap`.
        public unsafe Const_MapOrHashMap_MREdgeId_MREdgeId(MR._ByValue_MapOrHashMap_MREdgeId_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::createMap`.
        /// Parameter `size` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MREdgeId_MREdgeId> CreateMap(ulong? size = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_createMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_createMap(ulong *size);
            ulong __deref_size = size.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MREdgeId_MREdgeId(__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_createMap(size.HasValue ? &__deref_size : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::createHashMap`.
        /// Parameter `capacity` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MREdgeId_MREdgeId> CreateHashMap(ulong? capacity = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_createHashMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_createHashMap(ulong *capacity);
            ulong __deref_capacity = capacity.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MREdgeId_MREdgeId(__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_createHashMap(capacity.HasValue ? &__deref_capacity : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::getMap`.
        public unsafe MR.Const_EdgeMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getMap_const", ExactSpelling = true)]
            extern static MR.Const_EdgeMap._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_EdgeMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::getHashMap`.
        public unsafe MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getHashMap_const", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getHashMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getHashMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId(__ret, is_owning: false) : null;
        }
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>`.
    /// This is the non-const half of the class.
    public class MapOrHashMap_MREdgeId_MREdgeId : Const_MapOrHashMap_MREdgeId_MREdgeId
    {
        internal unsafe MapOrHashMap_MREdgeId_MREdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // default construction will select dense map
        public new unsafe MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_GetMutable_var", ExactSpelling = true)]
                extern static MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_GetMutable_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_GetMutable_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MapOrHashMap_MREdgeId_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>` elementwise.
        public unsafe MapOrHashMap_MREdgeId_MREdgeId(MR.Std._ByValue_Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MREdgeMap_PhmapFlatHashMapMREdgeIdMREdgeId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::MapOrHashMap`.
        public unsafe MapOrHashMap_MREdgeId_MREdgeId(MR._ByValue_MapOrHashMap_MREdgeId_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::operator=`.
        public unsafe MR.MapOrHashMap_MREdgeId_MREdgeId Assign(MR._ByValue_MapOrHashMap_MREdgeId_MREdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MREdgeId_MREdgeId._Underlying *_other);
            return new(__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::setMap`.
        public unsafe void SetMap(MR.Misc._Moved<MR.EdgeMap> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_setMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_setMap(_Underlying *_this, MR.EdgeMap._Underlying *m);
            __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_setMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::setHashMap`.
        public unsafe void SetHashMap(MR.Misc._Moved<MR.Phmap.FlatHashMap_MREdgeId_MREdgeId> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_setHashMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_setHashMap(_Underlying *_this, MR.Phmap.FlatHashMap_MREdgeId_MREdgeId._Underlying *m);
            __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_setHashMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// if this stores dense map then resizes it to denseTotalSize;
        /// if this stores hash map then sets its capacity to size()+hashAdditionalCapacity
        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::resizeReserve`.
        public unsafe void ResizeReserve(ulong denseTotalSize, ulong hashAdditionalCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_resizeReserve", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_resizeReserve(_Underlying *_this, ulong denseTotalSize, ulong hashAdditionalCapacity);
            __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_resizeReserve(_UnderlyingPtr, denseTotalSize, hashAdditionalCapacity);
        }

        /// appends one element in the map,
        /// in case of dense map, key must be equal to vector.endId()
        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::pushBack`.
        public unsafe void PushBack(MR.EdgeId key, MR.EdgeId val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_pushBack", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_pushBack(_Underlying *_this, MR.EdgeId key, MR.EdgeId val);
            __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_pushBack(_UnderlyingPtr, key, val);
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::getMap`.
        public unsafe new MR.EdgeMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getMap", ExactSpelling = true)]
            extern static MR.EdgeMap._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getMap(_UnderlyingPtr);
            return __ret is not null ? new MR.EdgeMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::getHashMap`.
        public unsafe new MR.Phmap.FlatHashMap_MREdgeId_MREdgeId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getHashMap", ExactSpelling = true)]
            extern static MR.Phmap.FlatHashMap_MREdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getHashMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_getHashMap(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.FlatHashMap_MREdgeId_MREdgeId(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::EdgeId, MR::EdgeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_clear", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_clear(_Underlying *_this);
            __MR_MapOrHashMap_MR_EdgeId_MR_EdgeId_clear(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MapOrHashMap_MREdgeId_MREdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MapOrHashMap_MREdgeId_MREdgeId`/`Const_MapOrHashMap_MREdgeId_MREdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MapOrHashMap_MREdgeId_MREdgeId
    {
        internal readonly Const_MapOrHashMap_MREdgeId_MREdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MapOrHashMap_MREdgeId_MREdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MapOrHashMap_MREdgeId_MREdgeId(Const_MapOrHashMap_MREdgeId_MREdgeId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MapOrHashMap_MREdgeId_MREdgeId(Const_MapOrHashMap_MREdgeId_MREdgeId arg) {return new(arg);}
        public _ByValue_MapOrHashMap_MREdgeId_MREdgeId(MR.Misc._Moved<MapOrHashMap_MREdgeId_MREdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MapOrHashMap_MREdgeId_MREdgeId(MR.Misc._Moved<MapOrHashMap_MREdgeId_MREdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MREdgeId_MREdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MapOrHashMap_MREdgeId_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MREdgeId_MREdgeId`/`Const_MapOrHashMap_MREdgeId_MREdgeId` directly.
    public class _InOptMut_MapOrHashMap_MREdgeId_MREdgeId
    {
        public MapOrHashMap_MREdgeId_MREdgeId? Opt;

        public _InOptMut_MapOrHashMap_MREdgeId_MREdgeId() {}
        public _InOptMut_MapOrHashMap_MREdgeId_MREdgeId(MapOrHashMap_MREdgeId_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_MapOrHashMap_MREdgeId_MREdgeId(MapOrHashMap_MREdgeId_MREdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MREdgeId_MREdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MapOrHashMap_MREdgeId_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MREdgeId_MREdgeId`/`Const_MapOrHashMap_MREdgeId_MREdgeId` to pass it to the function.
    public class _InOptConst_MapOrHashMap_MREdgeId_MREdgeId
    {
        public Const_MapOrHashMap_MREdgeId_MREdgeId? Opt;

        public _InOptConst_MapOrHashMap_MREdgeId_MREdgeId() {}
        public _InOptConst_MapOrHashMap_MREdgeId_MREdgeId(Const_MapOrHashMap_MREdgeId_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_MapOrHashMap_MREdgeId_MREdgeId(Const_MapOrHashMap_MREdgeId_MREdgeId value) {return new(value);}
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
    /// This is the const half of the class.
    public class Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy(_Underlying *_this);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId() {Dispose(false);}

        // default construction will select dense map
        public unsafe MR.Std.Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_var", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Get_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>` elementwise.
        public unsafe Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(MR.Std._ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::MapOrHashMap`.
        public unsafe Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(MR._ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::createMap`.
        /// Parameter `size` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId> CreateMap(ulong? size = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_createMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_createMap(ulong *size);
            ulong __deref_size = size.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_createMap(size.HasValue ? &__deref_size : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::createHashMap`.
        /// Parameter `capacity` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId> CreateHashMap(ulong? capacity = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_createHashMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_createHashMap(ulong *capacity);
            ulong __deref_capacity = capacity.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_createHashMap(capacity.HasValue ? &__deref_capacity : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::getMap`.
        public unsafe MR.Const_UndirectedEdgeMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getMap_const", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeMap._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_UndirectedEdgeMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::getHashMap`.
        public unsafe MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getHashMap_const", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getHashMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getHashMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(__ret, is_owning: false) : null;
        }
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
    /// This is the non-const half of the class.
    public class MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId : Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId
    {
        internal unsafe MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // default construction will select dense map
        public new unsafe MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_var", ExactSpelling = true)]
                extern static MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_GetMutable_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>` elementwise.
        public unsafe MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(MR.Std._ByValue_Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MRUndirectedEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMRUndirectedEdgeId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::MapOrHashMap`.
        public unsafe MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(MR._ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::operator=`.
        public unsafe MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId Assign(MR._ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *_other);
            return new(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::setMap`.
        public unsafe void SetMap(MR.Misc._Moved<MR.UndirectedEdgeMap> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_setMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_setMap(_Underlying *_this, MR.UndirectedEdgeMap._Underlying *m);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_setMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::setHashMap`.
        public unsafe void SetHashMap(MR.Misc._Moved<MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_setHashMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_setHashMap(_Underlying *_this, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *m);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_setHashMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// if this stores dense map then resizes it to denseTotalSize;
        /// if this stores hash map then sets its capacity to size()+hashAdditionalCapacity
        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::resizeReserve`.
        public unsafe void ResizeReserve(ulong denseTotalSize, ulong hashAdditionalCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_resizeReserve", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_resizeReserve(_Underlying *_this, ulong denseTotalSize, ulong hashAdditionalCapacity);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_resizeReserve(_UnderlyingPtr, denseTotalSize, hashAdditionalCapacity);
        }

        /// appends one element in the map,
        /// in case of dense map, key must be equal to vector.endId()
        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::pushBack`.
        public unsafe void PushBack(MR.UndirectedEdgeId key, MR.UndirectedEdgeId val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_pushBack", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_pushBack(_Underlying *_this, MR.UndirectedEdgeId key, MR.UndirectedEdgeId val);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_pushBack(_UnderlyingPtr, key, val);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::getMap`.
        public unsafe new MR.UndirectedEdgeMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getMap", ExactSpelling = true)]
            extern static MR.UndirectedEdgeMap._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getMap(_UnderlyingPtr);
            return __ret is not null ? new MR.UndirectedEdgeMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::getHashMap`.
        public unsafe new MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getHashMap", ExactSpelling = true)]
            extern static MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getHashMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_getHashMap(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_clear", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_clear(_Underlying *_this);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_UndirectedEdgeId_clear(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId`/`Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId
    {
        internal readonly Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId arg) {return new(arg);}
        public _ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(MR.Misc._Moved<MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(MR.Misc._Moved<MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId`/`Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId` directly.
    public class _InOptMut_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId
    {
        public MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? Opt;

        public _InOptMut_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId() {}
        public _InOptMut_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId`/`Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId` to pass it to the function.
    public class _InOptConst_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId
    {
        public Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? Opt;

        public _InOptConst_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId() {}
        public _InOptConst_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(Const_MapOrHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId value) {return new(value);}
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>`.
    /// This is the const half of the class.
    public class Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_Destroy(_Underlying *_this);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId() {Dispose(false);}

        // default construction will select dense map
        public unsafe MR.Std.Const_Variant_MRWholeEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMREdgeId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_Get_var", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRWholeEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_Get_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_Get_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>` elementwise.
        public unsafe Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(MR.Std._ByValue_Variant_MRWholeEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMREdgeId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MRWholeEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMREdgeId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::MapOrHashMap`.
        public unsafe Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(MR._ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::createMap`.
        /// Parameter `size` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId> CreateMap(ulong? size = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_createMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_createMap(ulong *size);
            ulong __deref_size = size.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_createMap(size.HasValue ? &__deref_size : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::createHashMap`.
        /// Parameter `capacity` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId> CreateHashMap(ulong? capacity = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_createHashMap", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_createHashMap(ulong *capacity);
            ulong __deref_capacity = capacity.GetValueOrDefault();
            return MR.Misc.Move(new MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_createHashMap(capacity.HasValue ? &__deref_capacity : null), is_owning: true));
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::getMap`.
        public unsafe MR.Const_WholeEdgeMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getMap_const", ExactSpelling = true)]
            extern static MR.Const_WholeEdgeMap._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_WholeEdgeMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::getHashMap`.
        public unsafe MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getHashMap_const", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getHashMap_const(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getHashMap_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId(__ret, is_owning: false) : null;
        }
    }

    /// stores a mapping from keys K to values V in one of two forms:
    /// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
    /// 2) as hash map preferable when valid keys are a small subset of the range
    /// Generated from class `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>`.
    /// This is the non-const half of the class.
    public class MapOrHashMap_MRUndirectedEdgeId_MREdgeId : Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId
    {
        internal unsafe MapOrHashMap_MRUndirectedEdgeId_MREdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // default construction will select dense map
        public new unsafe MR.Std.Variant_MRWholeEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMREdgeId Var
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_GetMutable_var", ExactSpelling = true)]
                extern static MR.Std.Variant_MRWholeEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_GetMutable_var(_Underlying *_this);
                return new(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_GetMutable_var(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MapOrHashMap_MRUndirectedEdgeId_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_DefaultConstruct();
        }

        /// Constructs `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>` elementwise.
        public unsafe MapOrHashMap_MRUndirectedEdgeId_MREdgeId(MR.Std._ByValue_Variant_MRWholeEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMREdgeId var) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFrom(MR.Misc._PassBy var_pass_by, MR.Std.Variant_MRWholeEdgeMap_PhmapFlatHashMapMRUndirectedEdgeIdMREdgeId._Underlying *var);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFrom(var.PassByMode, var.Value is not null ? var.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::MapOrHashMap`.
        public unsafe MapOrHashMap_MRUndirectedEdgeId_MREdgeId(MR._ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::operator=`.
        public unsafe MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId Assign(MR._ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *_other);
            return new(__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::setMap`.
        public unsafe void SetMap(MR.Misc._Moved<MR.WholeEdgeMap> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_setMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_setMap(_Underlying *_this, MR.WholeEdgeMap._Underlying *m);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_setMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::setHashMap`.
        public unsafe void SetHashMap(MR.Misc._Moved<MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MREdgeId> m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_setHashMap", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_setHashMap(_Underlying *_this, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *m);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_setHashMap(_UnderlyingPtr, m.Value._UnderlyingPtr);
        }

        /// if this stores dense map then resizes it to denseTotalSize;
        /// if this stores hash map then sets its capacity to size()+hashAdditionalCapacity
        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::resizeReserve`.
        public unsafe void ResizeReserve(ulong denseTotalSize, ulong hashAdditionalCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_resizeReserve", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_resizeReserve(_Underlying *_this, ulong denseTotalSize, ulong hashAdditionalCapacity);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_resizeReserve(_UnderlyingPtr, denseTotalSize, hashAdditionalCapacity);
        }

        /// appends one element in the map,
        /// in case of dense map, key must be equal to vector.endId()
        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::pushBack`.
        public unsafe void PushBack(MR.UndirectedEdgeId key, MR.EdgeId val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_pushBack", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_pushBack(_Underlying *_this, MR.UndirectedEdgeId key, MR.EdgeId val);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_pushBack(_UnderlyingPtr, key, val);
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::getMap`.
        public unsafe new MR.WholeEdgeMap? GetMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getMap", ExactSpelling = true)]
            extern static MR.WholeEdgeMap._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getMap(_UnderlyingPtr);
            return __ret is not null ? new MR.WholeEdgeMap(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::getHashMap`.
        public unsafe new MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MREdgeId? GetHashMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getHashMap", ExactSpelling = true)]
            extern static MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *__MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getHashMap(_Underlying *_this);
            var __ret = __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_getHashMap(_UnderlyingPtr);
            return __ret is not null ? new MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MREdgeId(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MapOrHashMap<MR::UndirectedEdgeId, MR::EdgeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_clear", ExactSpelling = true)]
            extern static void __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_clear(_Underlying *_this);
            __MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_clear(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MapOrHashMap_MRUndirectedEdgeId_MREdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MapOrHashMap_MRUndirectedEdgeId_MREdgeId`/`Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId
    {
        internal readonly Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId arg) {return new(arg);}
        public _ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(MR.Misc._Moved<MapOrHashMap_MRUndirectedEdgeId_MREdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(MR.Misc._Moved<MapOrHashMap_MRUndirectedEdgeId_MREdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MRUndirectedEdgeId_MREdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MapOrHashMap_MRUndirectedEdgeId_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MRUndirectedEdgeId_MREdgeId`/`Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId` directly.
    public class _InOptMut_MapOrHashMap_MRUndirectedEdgeId_MREdgeId
    {
        public MapOrHashMap_MRUndirectedEdgeId_MREdgeId? Opt;

        public _InOptMut_MapOrHashMap_MRUndirectedEdgeId_MREdgeId() {}
        public _InOptMut_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(MapOrHashMap_MRUndirectedEdgeId_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(MapOrHashMap_MRUndirectedEdgeId_MREdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `MapOrHashMap_MRUndirectedEdgeId_MREdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MapOrHashMap_MRUndirectedEdgeId_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MapOrHashMap_MRUndirectedEdgeId_MREdgeId`/`Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId` to pass it to the function.
    public class _InOptConst_MapOrHashMap_MRUndirectedEdgeId_MREdgeId
    {
        public Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId? Opt;

        public _InOptConst_MapOrHashMap_MRUndirectedEdgeId_MREdgeId() {}
        public _InOptConst_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_MapOrHashMap_MRUndirectedEdgeId_MREdgeId(Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId value) {return new(value);}
    }
}

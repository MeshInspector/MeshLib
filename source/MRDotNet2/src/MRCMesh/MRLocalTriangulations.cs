public static partial class MR
{
    /// describes one fan of triangles around a point excluding the point
    /// Generated from class `MR::FanRecord`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::FanRecordWithCenter`
    /// This is the const half of the class.
    public class Const_FanRecord : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FanRecord(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_Destroy", ExactSpelling = true)]
            extern static void __MR_FanRecord_Destroy(_Underlying *_this);
            __MR_FanRecord_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FanRecord() {Dispose(false);}

        /// first border edge (invalid if the center point is not on the boundary);
        /// triangle associated with this point is absent
        public unsafe MR.Const_VertId Border
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_Get_border", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_FanRecord_Get_border(_Underlying *_this);
                return new(__MR_FanRecord_Get_border(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the position of first neigbor in LocalTriangulations::neighbours
        public unsafe uint FirstNei
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_Get_firstNei", ExactSpelling = true)]
                extern static uint *__MR_FanRecord_Get_firstNei(_Underlying *_this);
                return *__MR_FanRecord_Get_firstNei(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        /// Parameter `b` defaults to `{}`.
        /// Parameter `fn` defaults to `0`.
        public unsafe Const_FanRecord(MR._InOpt_VertId b = default, uint? fn = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_Construct_2", ExactSpelling = true)]
            extern static MR.FanRecord._Underlying *__MR_FanRecord_Construct_2(MR.VertId *b, uint *fn);
            uint __deref_fn = fn.GetValueOrDefault();
            _UnderlyingPtr = __MR_FanRecord_Construct_2(b.HasValue ? &b.Object : null, fn.HasValue ? &__deref_fn : null);
        }

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        public unsafe Const_FanRecord(MR.Const_FanRecord _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FanRecord._Underlying *__MR_FanRecord_ConstructFromAnother(MR.FanRecord._Underlying *_other);
            _UnderlyingPtr = __MR_FanRecord_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        public unsafe Const_FanRecord(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_Construct_1", ExactSpelling = true)]
            extern static MR.FanRecord._Underlying *__MR_FanRecord_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_FanRecord_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        public static unsafe implicit operator Const_FanRecord(MR.Const_NoInit _1) {return new(_1);}
    }

    /// describes one fan of triangles around a point excluding the point
    /// Generated from class `MR::FanRecord`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::FanRecordWithCenter`
    /// This is the non-const half of the class.
    public class FanRecord : Const_FanRecord
    {
        internal unsafe FanRecord(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// first border edge (invalid if the center point is not on the boundary);
        /// triangle associated with this point is absent
        public new unsafe MR.Mut_VertId Border
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_GetMutable_border", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_FanRecord_GetMutable_border(_Underlying *_this);
                return new(__MR_FanRecord_GetMutable_border(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the position of first neigbor in LocalTriangulations::neighbours
        public new unsafe ref uint FirstNei
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_GetMutable_firstNei", ExactSpelling = true)]
                extern static uint *__MR_FanRecord_GetMutable_firstNei(_Underlying *_this);
                return ref *__MR_FanRecord_GetMutable_firstNei(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        /// Parameter `b` defaults to `{}`.
        /// Parameter `fn` defaults to `0`.
        public unsafe FanRecord(MR._InOpt_VertId b = default, uint? fn = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_Construct_2", ExactSpelling = true)]
            extern static MR.FanRecord._Underlying *__MR_FanRecord_Construct_2(MR.VertId *b, uint *fn);
            uint __deref_fn = fn.GetValueOrDefault();
            _UnderlyingPtr = __MR_FanRecord_Construct_2(b.HasValue ? &b.Object : null, fn.HasValue ? &__deref_fn : null);
        }

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        public unsafe FanRecord(MR.Const_FanRecord _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FanRecord._Underlying *__MR_FanRecord_ConstructFromAnother(MR.FanRecord._Underlying *_other);
            _UnderlyingPtr = __MR_FanRecord_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        public unsafe FanRecord(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_Construct_1", ExactSpelling = true)]
            extern static MR.FanRecord._Underlying *__MR_FanRecord_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_FanRecord_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        public static unsafe implicit operator FanRecord(MR.Const_NoInit _1) {return new(_1);}

        /// Generated from method `MR::FanRecord::operator=`.
        public unsafe MR.FanRecord Assign(MR.Const_FanRecord _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecord_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FanRecord._Underlying *__MR_FanRecord_AssignFromAnother(_Underlying *_this, MR.FanRecord._Underlying *_other);
            return new(__MR_FanRecord_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FanRecord` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FanRecord`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FanRecord`/`Const_FanRecord` directly.
    public class _InOptMut_FanRecord
    {
        public FanRecord? Opt;

        public _InOptMut_FanRecord() {}
        public _InOptMut_FanRecord(FanRecord value) {Opt = value;}
        public static implicit operator _InOptMut_FanRecord(FanRecord value) {return new(value);}
    }

    /// This is used for optional parameters of class `FanRecord` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FanRecord`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FanRecord`/`Const_FanRecord` to pass it to the function.
    public class _InOptConst_FanRecord
    {
        public Const_FanRecord? Opt;

        public _InOptConst_FanRecord() {}
        public _InOptConst_FanRecord(Const_FanRecord value) {Opt = value;}
        public static implicit operator _InOptConst_FanRecord(Const_FanRecord value) {return new(value);}

        /// Generated from constructor `MR::FanRecord::FanRecord`.
        public static unsafe implicit operator _InOptConst_FanRecord(MR.Const_NoInit _1) {return new MR.FanRecord(_1);}
    }

    /// describes one fan of triangles around a point including the point
    /// Generated from class `MR::FanRecordWithCenter`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FanRecord`
    /// This is the const half of the class.
    public class Const_FanRecordWithCenter : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FanRecordWithCenter(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_Destroy", ExactSpelling = true)]
            extern static void __MR_FanRecordWithCenter_Destroy(_Underlying *_this);
            __MR_FanRecordWithCenter_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FanRecordWithCenter() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_FanRecord(Const_FanRecordWithCenter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_UpcastTo_MR_FanRecord", ExactSpelling = true)]
            extern static MR.Const_FanRecord._Underlying *__MR_FanRecordWithCenter_UpcastTo_MR_FanRecord(_Underlying *_this);
            MR.Const_FanRecord ret = new(__MR_FanRecordWithCenter_UpcastTo_MR_FanRecord(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// center point in the fan
        public unsafe MR.Const_VertId Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_Get_center", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_FanRecordWithCenter_Get_center(_Underlying *_this);
                return new(__MR_FanRecordWithCenter_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        /// first border edge (invalid if the center point is not on the boundary);
        /// triangle associated with this point is absent
        public unsafe MR.Const_VertId Border
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_Get_border", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_FanRecordWithCenter_Get_border(_Underlying *_this);
                return new(__MR_FanRecordWithCenter_Get_border(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the position of first neigbor in LocalTriangulations::neighbours
        public unsafe uint FirstNei
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_Get_firstNei", ExactSpelling = true)]
                extern static uint *__MR_FanRecordWithCenter_Get_firstNei(_Underlying *_this);
                return *__MR_FanRecordWithCenter_Get_firstNei(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        /// Parameter `c` defaults to `{}`.
        /// Parameter `b` defaults to `{}`.
        /// Parameter `fn` defaults to `0`.
        public unsafe Const_FanRecordWithCenter(MR._InOpt_VertId c = default, MR._InOpt_VertId b = default, uint? fn = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_Construct_3", ExactSpelling = true)]
            extern static MR.FanRecordWithCenter._Underlying *__MR_FanRecordWithCenter_Construct_3(MR.VertId *c, MR.VertId *b, uint *fn);
            uint __deref_fn = fn.GetValueOrDefault();
            _UnderlyingPtr = __MR_FanRecordWithCenter_Construct_3(c.HasValue ? &c.Object : null, b.HasValue ? &b.Object : null, fn.HasValue ? &__deref_fn : null);
        }

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        public unsafe Const_FanRecordWithCenter(MR.Const_FanRecordWithCenter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FanRecordWithCenter._Underlying *__MR_FanRecordWithCenter_ConstructFromAnother(MR.FanRecordWithCenter._Underlying *_other);
            _UnderlyingPtr = __MR_FanRecordWithCenter_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        public unsafe Const_FanRecordWithCenter(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_Construct_1", ExactSpelling = true)]
            extern static MR.FanRecordWithCenter._Underlying *__MR_FanRecordWithCenter_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_FanRecordWithCenter_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        public static unsafe implicit operator Const_FanRecordWithCenter(MR.Const_NoInit _1) {return new(_1);}
    }

    /// describes one fan of triangles around a point including the point
    /// Generated from class `MR::FanRecordWithCenter`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FanRecord`
    /// This is the non-const half of the class.
    public class FanRecordWithCenter : Const_FanRecordWithCenter
    {
        internal unsafe FanRecordWithCenter(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.FanRecord(FanRecordWithCenter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_UpcastTo_MR_FanRecord", ExactSpelling = true)]
            extern static MR.FanRecord._Underlying *__MR_FanRecordWithCenter_UpcastTo_MR_FanRecord(_Underlying *_this);
            MR.FanRecord ret = new(__MR_FanRecordWithCenter_UpcastTo_MR_FanRecord(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// center point in the fan
        public new unsafe MR.Mut_VertId Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_FanRecordWithCenter_GetMutable_center(_Underlying *_this);
                return new(__MR_FanRecordWithCenter_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        /// first border edge (invalid if the center point is not on the boundary);
        /// triangle associated with this point is absent
        public new unsafe MR.Mut_VertId Border
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_GetMutable_border", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_FanRecordWithCenter_GetMutable_border(_Underlying *_this);
                return new(__MR_FanRecordWithCenter_GetMutable_border(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the position of first neigbor in LocalTriangulations::neighbours
        public new unsafe ref uint FirstNei
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_GetMutable_firstNei", ExactSpelling = true)]
                extern static uint *__MR_FanRecordWithCenter_GetMutable_firstNei(_Underlying *_this);
                return ref *__MR_FanRecordWithCenter_GetMutable_firstNei(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        /// Parameter `c` defaults to `{}`.
        /// Parameter `b` defaults to `{}`.
        /// Parameter `fn` defaults to `0`.
        public unsafe FanRecordWithCenter(MR._InOpt_VertId c = default, MR._InOpt_VertId b = default, uint? fn = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_Construct_3", ExactSpelling = true)]
            extern static MR.FanRecordWithCenter._Underlying *__MR_FanRecordWithCenter_Construct_3(MR.VertId *c, MR.VertId *b, uint *fn);
            uint __deref_fn = fn.GetValueOrDefault();
            _UnderlyingPtr = __MR_FanRecordWithCenter_Construct_3(c.HasValue ? &c.Object : null, b.HasValue ? &b.Object : null, fn.HasValue ? &__deref_fn : null);
        }

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        public unsafe FanRecordWithCenter(MR.Const_FanRecordWithCenter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FanRecordWithCenter._Underlying *__MR_FanRecordWithCenter_ConstructFromAnother(MR.FanRecordWithCenter._Underlying *_other);
            _UnderlyingPtr = __MR_FanRecordWithCenter_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        public unsafe FanRecordWithCenter(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_Construct_1", ExactSpelling = true)]
            extern static MR.FanRecordWithCenter._Underlying *__MR_FanRecordWithCenter_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_FanRecordWithCenter_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        public static unsafe implicit operator FanRecordWithCenter(MR.Const_NoInit _1) {return new(_1);}

        /// Generated from method `MR::FanRecordWithCenter::operator=`.
        public unsafe MR.FanRecordWithCenter Assign(MR.Const_FanRecordWithCenter _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FanRecordWithCenter_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FanRecordWithCenter._Underlying *__MR_FanRecordWithCenter_AssignFromAnother(_Underlying *_this, MR.FanRecordWithCenter._Underlying *_other);
            return new(__MR_FanRecordWithCenter_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FanRecordWithCenter` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FanRecordWithCenter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FanRecordWithCenter`/`Const_FanRecordWithCenter` directly.
    public class _InOptMut_FanRecordWithCenter
    {
        public FanRecordWithCenter? Opt;

        public _InOptMut_FanRecordWithCenter() {}
        public _InOptMut_FanRecordWithCenter(FanRecordWithCenter value) {Opt = value;}
        public static implicit operator _InOptMut_FanRecordWithCenter(FanRecordWithCenter value) {return new(value);}
    }

    /// This is used for optional parameters of class `FanRecordWithCenter` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FanRecordWithCenter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FanRecordWithCenter`/`Const_FanRecordWithCenter` to pass it to the function.
    public class _InOptConst_FanRecordWithCenter
    {
        public Const_FanRecordWithCenter? Opt;

        public _InOptConst_FanRecordWithCenter() {}
        public _InOptConst_FanRecordWithCenter(Const_FanRecordWithCenter value) {Opt = value;}
        public static implicit operator _InOptConst_FanRecordWithCenter(Const_FanRecordWithCenter value) {return new(value);}

        /// Generated from constructor `MR::FanRecordWithCenter::FanRecordWithCenter`.
        public static unsafe implicit operator _InOptConst_FanRecordWithCenter(MR.Const_NoInit _1) {return new MR.FanRecordWithCenter(_1);}
    }

    /// describes a number of local triangulations of some points (e.g. assigned to a thread)
    /// Generated from class `MR::SomeLocalTriangulations`.
    /// This is the const half of the class.
    public class Const_SomeLocalTriangulations : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SomeLocalTriangulations(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_Destroy", ExactSpelling = true)]
            extern static void __MR_SomeLocalTriangulations_Destroy(_Underlying *_this);
            __MR_SomeLocalTriangulations_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SomeLocalTriangulations() {Dispose(false);}

        public unsafe MR.Std.Const_Vector_MRVertId Neighbors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_Get_neighbors", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRVertId._Underlying *__MR_SomeLocalTriangulations_Get_neighbors(_Underlying *_this);
                return new(__MR_SomeLocalTriangulations_Get_neighbors(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Vector_MRFanRecordWithCenter FanRecords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_Get_fanRecords", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRFanRecordWithCenter._Underlying *__MR_SomeLocalTriangulations_Get_fanRecords(_Underlying *_this);
                return new(__MR_SomeLocalTriangulations_Get_fanRecords(_UnderlyingPtr), is_owning: false);
            }
        }

        //in fanRecords
        public unsafe MR.Const_VertId MaxCenterId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_Get_maxCenterId", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_SomeLocalTriangulations_Get_maxCenterId(_Underlying *_this);
                return new(__MR_SomeLocalTriangulations_Get_maxCenterId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SomeLocalTriangulations() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SomeLocalTriangulations._Underlying *__MR_SomeLocalTriangulations_DefaultConstruct();
            _UnderlyingPtr = __MR_SomeLocalTriangulations_DefaultConstruct();
        }

        /// Constructs `MR::SomeLocalTriangulations` elementwise.
        public unsafe Const_SomeLocalTriangulations(MR.Std._ByValue_Vector_MRVertId neighbors, MR.Std._ByValue_Vector_MRFanRecordWithCenter fanRecords, MR.VertId maxCenterId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_ConstructFrom", ExactSpelling = true)]
            extern static MR.SomeLocalTriangulations._Underlying *__MR_SomeLocalTriangulations_ConstructFrom(MR.Misc._PassBy neighbors_pass_by, MR.Std.Vector_MRVertId._Underlying *neighbors, MR.Misc._PassBy fanRecords_pass_by, MR.Std.Vector_MRFanRecordWithCenter._Underlying *fanRecords, MR.VertId maxCenterId);
            _UnderlyingPtr = __MR_SomeLocalTriangulations_ConstructFrom(neighbors.PassByMode, neighbors.Value is not null ? neighbors.Value._UnderlyingPtr : null, fanRecords.PassByMode, fanRecords.Value is not null ? fanRecords.Value._UnderlyingPtr : null, maxCenterId);
        }

        /// Generated from constructor `MR::SomeLocalTriangulations::SomeLocalTriangulations`.
        public unsafe Const_SomeLocalTriangulations(MR._ByValue_SomeLocalTriangulations _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SomeLocalTriangulations._Underlying *__MR_SomeLocalTriangulations_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SomeLocalTriangulations._Underlying *_other);
            _UnderlyingPtr = __MR_SomeLocalTriangulations_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// describes a number of local triangulations of some points (e.g. assigned to a thread)
    /// Generated from class `MR::SomeLocalTriangulations`.
    /// This is the non-const half of the class.
    public class SomeLocalTriangulations : Const_SomeLocalTriangulations
    {
        internal unsafe SomeLocalTriangulations(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Vector_MRVertId Neighbors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_GetMutable_neighbors", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVertId._Underlying *__MR_SomeLocalTriangulations_GetMutable_neighbors(_Underlying *_this);
                return new(__MR_SomeLocalTriangulations_GetMutable_neighbors(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Vector_MRFanRecordWithCenter FanRecords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_GetMutable_fanRecords", ExactSpelling = true)]
                extern static MR.Std.Vector_MRFanRecordWithCenter._Underlying *__MR_SomeLocalTriangulations_GetMutable_fanRecords(_Underlying *_this);
                return new(__MR_SomeLocalTriangulations_GetMutable_fanRecords(_UnderlyingPtr), is_owning: false);
            }
        }

        //in fanRecords
        public new unsafe MR.Mut_VertId MaxCenterId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_GetMutable_maxCenterId", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_SomeLocalTriangulations_GetMutable_maxCenterId(_Underlying *_this);
                return new(__MR_SomeLocalTriangulations_GetMutable_maxCenterId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SomeLocalTriangulations() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SomeLocalTriangulations._Underlying *__MR_SomeLocalTriangulations_DefaultConstruct();
            _UnderlyingPtr = __MR_SomeLocalTriangulations_DefaultConstruct();
        }

        /// Constructs `MR::SomeLocalTriangulations` elementwise.
        public unsafe SomeLocalTriangulations(MR.Std._ByValue_Vector_MRVertId neighbors, MR.Std._ByValue_Vector_MRFanRecordWithCenter fanRecords, MR.VertId maxCenterId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_ConstructFrom", ExactSpelling = true)]
            extern static MR.SomeLocalTriangulations._Underlying *__MR_SomeLocalTriangulations_ConstructFrom(MR.Misc._PassBy neighbors_pass_by, MR.Std.Vector_MRVertId._Underlying *neighbors, MR.Misc._PassBy fanRecords_pass_by, MR.Std.Vector_MRFanRecordWithCenter._Underlying *fanRecords, MR.VertId maxCenterId);
            _UnderlyingPtr = __MR_SomeLocalTriangulations_ConstructFrom(neighbors.PassByMode, neighbors.Value is not null ? neighbors.Value._UnderlyingPtr : null, fanRecords.PassByMode, fanRecords.Value is not null ? fanRecords.Value._UnderlyingPtr : null, maxCenterId);
        }

        /// Generated from constructor `MR::SomeLocalTriangulations::SomeLocalTriangulations`.
        public unsafe SomeLocalTriangulations(MR._ByValue_SomeLocalTriangulations _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SomeLocalTriangulations._Underlying *__MR_SomeLocalTriangulations_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SomeLocalTriangulations._Underlying *_other);
            _UnderlyingPtr = __MR_SomeLocalTriangulations_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SomeLocalTriangulations::operator=`.
        public unsafe MR.SomeLocalTriangulations Assign(MR._ByValue_SomeLocalTriangulations _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SomeLocalTriangulations_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SomeLocalTriangulations._Underlying *__MR_SomeLocalTriangulations_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SomeLocalTriangulations._Underlying *_other);
            return new(__MR_SomeLocalTriangulations_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SomeLocalTriangulations` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SomeLocalTriangulations`/`Const_SomeLocalTriangulations` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SomeLocalTriangulations
    {
        internal readonly Const_SomeLocalTriangulations? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SomeLocalTriangulations() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SomeLocalTriangulations(Const_SomeLocalTriangulations new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SomeLocalTriangulations(Const_SomeLocalTriangulations arg) {return new(arg);}
        public _ByValue_SomeLocalTriangulations(MR.Misc._Moved<SomeLocalTriangulations> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SomeLocalTriangulations(MR.Misc._Moved<SomeLocalTriangulations> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SomeLocalTriangulations` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SomeLocalTriangulations`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SomeLocalTriangulations`/`Const_SomeLocalTriangulations` directly.
    public class _InOptMut_SomeLocalTriangulations
    {
        public SomeLocalTriangulations? Opt;

        public _InOptMut_SomeLocalTriangulations() {}
        public _InOptMut_SomeLocalTriangulations(SomeLocalTriangulations value) {Opt = value;}
        public static implicit operator _InOptMut_SomeLocalTriangulations(SomeLocalTriangulations value) {return new(value);}
    }

    /// This is used for optional parameters of class `SomeLocalTriangulations` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SomeLocalTriangulations`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SomeLocalTriangulations`/`Const_SomeLocalTriangulations` to pass it to the function.
    public class _InOptConst_SomeLocalTriangulations
    {
        public Const_SomeLocalTriangulations? Opt;

        public _InOptConst_SomeLocalTriangulations() {}
        public _InOptConst_SomeLocalTriangulations(Const_SomeLocalTriangulations value) {Opt = value;}
        public static implicit operator _InOptConst_SomeLocalTriangulations(Const_SomeLocalTriangulations value) {return new(value);}
    }

    /// triangulations for all points, with easy access by VertId
    /// Generated from class `MR::AllLocalTriangulations`.
    /// This is the const half of the class.
    public class Const_AllLocalTriangulations : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AllLocalTriangulations(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_Destroy", ExactSpelling = true)]
            extern static void __MR_AllLocalTriangulations_Destroy(_Underlying *_this);
            __MR_AllLocalTriangulations_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AllLocalTriangulations() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRVertId Neighbors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_Get_neighbors", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRVertId._Underlying *__MR_AllLocalTriangulations_Get_neighbors(_Underlying *_this);
                return new(__MR_AllLocalTriangulations_Get_neighbors(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector_MRFanRecord_MRVertId FanRecords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_Get_fanRecords", ExactSpelling = true)]
                extern static MR.Const_Vector_MRFanRecord_MRVertId._Underlying *__MR_AllLocalTriangulations_Get_fanRecords(_Underlying *_this);
                return new(__MR_AllLocalTriangulations_Get_fanRecords(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AllLocalTriangulations() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AllLocalTriangulations._Underlying *__MR_AllLocalTriangulations_DefaultConstruct();
            _UnderlyingPtr = __MR_AllLocalTriangulations_DefaultConstruct();
        }

        /// Constructs `MR::AllLocalTriangulations` elementwise.
        public unsafe Const_AllLocalTriangulations(MR._ByValue_Buffer_MRVertId neighbors, MR._ByValue_Vector_MRFanRecord_MRVertId fanRecords) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_ConstructFrom", ExactSpelling = true)]
            extern static MR.AllLocalTriangulations._Underlying *__MR_AllLocalTriangulations_ConstructFrom(MR.Misc._PassBy neighbors_pass_by, MR.Buffer_MRVertId._Underlying *neighbors, MR.Misc._PassBy fanRecords_pass_by, MR.Vector_MRFanRecord_MRVertId._Underlying *fanRecords);
            _UnderlyingPtr = __MR_AllLocalTriangulations_ConstructFrom(neighbors.PassByMode, neighbors.Value is not null ? neighbors.Value._UnderlyingPtr : null, fanRecords.PassByMode, fanRecords.Value is not null ? fanRecords.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::AllLocalTriangulations::AllLocalTriangulations`.
        public unsafe Const_AllLocalTriangulations(MR._ByValue_AllLocalTriangulations _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AllLocalTriangulations._Underlying *__MR_AllLocalTriangulations_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AllLocalTriangulations._Underlying *_other);
            _UnderlyingPtr = __MR_AllLocalTriangulations_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// triangulations for all points, with easy access by VertId
    /// Generated from class `MR::AllLocalTriangulations`.
    /// This is the non-const half of the class.
    public class AllLocalTriangulations : Const_AllLocalTriangulations
    {
        internal unsafe AllLocalTriangulations(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRVertId Neighbors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_GetMutable_neighbors", ExactSpelling = true)]
                extern static MR.Buffer_MRVertId._Underlying *__MR_AllLocalTriangulations_GetMutable_neighbors(_Underlying *_this);
                return new(__MR_AllLocalTriangulations_GetMutable_neighbors(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Vector_MRFanRecord_MRVertId FanRecords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_GetMutable_fanRecords", ExactSpelling = true)]
                extern static MR.Vector_MRFanRecord_MRVertId._Underlying *__MR_AllLocalTriangulations_GetMutable_fanRecords(_Underlying *_this);
                return new(__MR_AllLocalTriangulations_GetMutable_fanRecords(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AllLocalTriangulations() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AllLocalTriangulations._Underlying *__MR_AllLocalTriangulations_DefaultConstruct();
            _UnderlyingPtr = __MR_AllLocalTriangulations_DefaultConstruct();
        }

        /// Constructs `MR::AllLocalTriangulations` elementwise.
        public unsafe AllLocalTriangulations(MR._ByValue_Buffer_MRVertId neighbors, MR._ByValue_Vector_MRFanRecord_MRVertId fanRecords) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_ConstructFrom", ExactSpelling = true)]
            extern static MR.AllLocalTriangulations._Underlying *__MR_AllLocalTriangulations_ConstructFrom(MR.Misc._PassBy neighbors_pass_by, MR.Buffer_MRVertId._Underlying *neighbors, MR.Misc._PassBy fanRecords_pass_by, MR.Vector_MRFanRecord_MRVertId._Underlying *fanRecords);
            _UnderlyingPtr = __MR_AllLocalTriangulations_ConstructFrom(neighbors.PassByMode, neighbors.Value is not null ? neighbors.Value._UnderlyingPtr : null, fanRecords.PassByMode, fanRecords.Value is not null ? fanRecords.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::AllLocalTriangulations::AllLocalTriangulations`.
        public unsafe AllLocalTriangulations(MR._ByValue_AllLocalTriangulations _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AllLocalTriangulations._Underlying *__MR_AllLocalTriangulations_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AllLocalTriangulations._Underlying *_other);
            _UnderlyingPtr = __MR_AllLocalTriangulations_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AllLocalTriangulations::operator=`.
        public unsafe MR.AllLocalTriangulations Assign(MR._ByValue_AllLocalTriangulations _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AllLocalTriangulations_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AllLocalTriangulations._Underlying *__MR_AllLocalTriangulations_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AllLocalTriangulations._Underlying *_other);
            return new(__MR_AllLocalTriangulations_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AllLocalTriangulations` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AllLocalTriangulations`/`Const_AllLocalTriangulations` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AllLocalTriangulations
    {
        internal readonly Const_AllLocalTriangulations? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AllLocalTriangulations() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AllLocalTriangulations(MR.Misc._Moved<AllLocalTriangulations> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AllLocalTriangulations(MR.Misc._Moved<AllLocalTriangulations> arg) {return new(arg);}
    }

    /// This is used as a function parameter when the underlying function receives an optional `AllLocalTriangulations` by value,
    ///   and also has a default argument, meaning it has two different null states.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AllLocalTriangulations`/`Const_AllLocalTriangulations` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument.
    /// * Pass `MR.Misc.NullOptType` to pass no object.
    public class _ByValueOptOpt_AllLocalTriangulations
    {
        internal readonly Const_AllLocalTriangulations? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValueOptOpt_AllLocalTriangulations() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValueOptOpt_AllLocalTriangulations(MR.Misc._Moved<AllLocalTriangulations> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValueOptOpt_AllLocalTriangulations(MR.Misc._Moved<AllLocalTriangulations> arg) {return new(arg);}
        public _ByValueOptOpt_AllLocalTriangulations(MR.Misc.NullOptType nullopt) {PassByMode = MR.Misc._PassBy.no_object;}
        public static implicit operator _ByValueOptOpt_AllLocalTriangulations(MR.Misc.NullOptType nullopt) {return new(nullopt);}
    }

    /// This is used for optional parameters of class `AllLocalTriangulations` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AllLocalTriangulations`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AllLocalTriangulations`/`Const_AllLocalTriangulations` directly.
    public class _InOptMut_AllLocalTriangulations
    {
        public AllLocalTriangulations? Opt;

        public _InOptMut_AllLocalTriangulations() {}
        public _InOptMut_AllLocalTriangulations(AllLocalTriangulations value) {Opt = value;}
        public static implicit operator _InOptMut_AllLocalTriangulations(AllLocalTriangulations value) {return new(value);}
    }

    /// This is used for optional parameters of class `AllLocalTriangulations` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AllLocalTriangulations`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AllLocalTriangulations`/`Const_AllLocalTriangulations` to pass it to the function.
    public class _InOptConst_AllLocalTriangulations
    {
        public Const_AllLocalTriangulations? Opt;

        public _InOptConst_AllLocalTriangulations() {}
        public _InOptConst_AllLocalTriangulations(Const_AllLocalTriangulations value) {Opt = value;}
        public static implicit operator _InOptConst_AllLocalTriangulations(Const_AllLocalTriangulations value) {return new(value);}
    }

    /// converts a set of SomeLocalTriangulations containing local triangulations of all points arbitrary distributed among them
    /// into one AllLocalTriangulations with records for all points
    /// Generated from function `MR::uniteLocalTriangulations`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRAllLocalTriangulations> UniteLocalTriangulations(MR.Std.Const_Vector_MRSomeLocalTriangulations in_, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_uniteLocalTriangulations", ExactSpelling = true)]
        extern static MR.Std.Optional_MRAllLocalTriangulations._Underlying *__MR_uniteLocalTriangulations(MR.Std.Const_Vector_MRSomeLocalTriangulations._Underlying *in_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.Std.Optional_MRAllLocalTriangulations(__MR_uniteLocalTriangulations(in_._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }

    /// compute normal at point by averaging neighbor triangle normals weighted by triangle's angle at the point
    /// Generated from function `MR::computeNormal`.
    public static unsafe MR.Vector3f ComputeNormal(MR.Const_AllLocalTriangulations triangs, MR.Const_VertCoords points, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeNormal", ExactSpelling = true)]
        extern static MR.Vector3f __MR_computeNormal(MR.Const_AllLocalTriangulations._Underlying *triangs, MR.Const_VertCoords._Underlying *points, MR.VertId v);
        return __MR_computeNormal(triangs._UnderlyingPtr, points._UnderlyingPtr, v);
    }

    /// orient neighbors around each point in \param region so they will be in clockwise order if look from the tip of target direction
    /// Generated from function `MR::orientLocalTriangulations`.
    public static unsafe void OrientLocalTriangulations(MR.AllLocalTriangulations triangs, MR.Const_VertCoords coords, MR.Const_VertBitSet region, MR.Const_VertCoords targetDir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orientLocalTriangulations_MR_VertCoords", ExactSpelling = true)]
        extern static void __MR_orientLocalTriangulations_MR_VertCoords(MR.AllLocalTriangulations._Underlying *triangs, MR.Const_VertCoords._Underlying *coords, MR.Const_VertBitSet._Underlying *region, MR.Const_VertCoords._Underlying *targetDir);
        __MR_orientLocalTriangulations_MR_VertCoords(triangs._UnderlyingPtr, coords._UnderlyingPtr, region._UnderlyingPtr, targetDir._UnderlyingPtr);
    }

    /// Generated from function `MR::orientLocalTriangulations`.
    public static unsafe void OrientLocalTriangulations(MR.AllLocalTriangulations triangs, MR.Const_VertCoords coords, MR.Const_VertBitSet region, MR.Std.Const_Function_MRVector3fFuncFromMRVertId targetDir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orientLocalTriangulations_std_function_MR_Vector3f_func_from_MR_VertId", ExactSpelling = true)]
        extern static void __MR_orientLocalTriangulations_std_function_MR_Vector3f_func_from_MR_VertId(MR.AllLocalTriangulations._Underlying *triangs, MR.Const_VertCoords._Underlying *coords, MR.Const_VertBitSet._Underlying *region, MR.Std.Const_Function_MRVector3fFuncFromMRVertId._Underlying *targetDir);
        __MR_orientLocalTriangulations_std_function_MR_Vector3f_func_from_MR_VertId(triangs._UnderlyingPtr, coords._UnderlyingPtr, region._UnderlyingPtr, targetDir._UnderlyingPtr);
    }

    /// orient neighbors around each point in \param region so there will be as many triangles with same (and not opposite) orientation as possible
    /// Generated from function `MR::autoOrientLocalTriangulations`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe bool AutoOrientLocalTriangulations(MR.Const_PointCloud pointCloud, MR.AllLocalTriangulations triangs, MR.Const_VertBitSet region, MR.Std._ByValue_Function_BoolFuncFromFloat? progress = null, MR.Triangulation? outRep3 = null, MR.Triangulation? outRep2 = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_autoOrientLocalTriangulations", ExactSpelling = true)]
        extern static byte __MR_autoOrientLocalTriangulations(MR.Const_PointCloud._Underlying *pointCloud, MR.AllLocalTriangulations._Underlying *triangs, MR.Const_VertBitSet._Underlying *region, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress, MR.Triangulation._Underlying *outRep3, MR.Triangulation._Underlying *outRep2);
        return __MR_autoOrientLocalTriangulations(pointCloud._UnderlyingPtr, triangs._UnderlyingPtr, region._UnderlyingPtr, progress is not null ? progress.PassByMode : MR.Misc._PassBy.default_arg, progress is not null && progress.Value is not null ? progress.Value._UnderlyingPtr : null, outRep3 is not null ? outRep3._UnderlyingPtr : null, outRep2 is not null ? outRep2._UnderlyingPtr : null) != 0;
    }

    /// computes statistics about the number of triangle repetitions in local triangulations
    /// Generated from function `MR::computeTrianglesRepetitions`.
    public static unsafe MR.Std.Array_Int_4 ComputeTrianglesRepetitions(MR.Const_AllLocalTriangulations triangs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeTrianglesRepetitions", ExactSpelling = true)]
        extern static MR.Std.Array_Int_4 __MR_computeTrianglesRepetitions(MR.Const_AllLocalTriangulations._Underlying *triangs);
        return __MR_computeTrianglesRepetitions(triangs._UnderlyingPtr);
    }

    /// from local triangulations returns all unoriented triangles with given number of repetitions each in [1,3]
    /// Generated from function `MR::findRepeatedUnorientedTriangles`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRUnorientedTriangle> FindRepeatedUnorientedTriangles(MR.Const_AllLocalTriangulations triangs, int repetitions)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findRepeatedUnorientedTriangles", ExactSpelling = true)]
        extern static MR.Std.Vector_MRUnorientedTriangle._Underlying *__MR_findRepeatedUnorientedTriangles(MR.Const_AllLocalTriangulations._Underlying *triangs, int repetitions);
        return MR.Misc.Move(new MR.Std.Vector_MRUnorientedTriangle(__MR_findRepeatedUnorientedTriangles(triangs._UnderlyingPtr, repetitions), is_owning: true));
    }

    /// from local triangulations returns all oriented triangles with given number of repetitions each in [1,3]
    /// Generated from function `MR::findRepeatedOrientedTriangles`.
    public static unsafe MR.Misc._Moved<MR.Triangulation> FindRepeatedOrientedTriangles(MR.Const_AllLocalTriangulations triangs, int repetitions)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findRepeatedOrientedTriangles_2", ExactSpelling = true)]
        extern static MR.Triangulation._Underlying *__MR_findRepeatedOrientedTriangles_2(MR.Const_AllLocalTriangulations._Underlying *triangs, int repetitions);
        return MR.Misc.Move(new MR.Triangulation(__MR_findRepeatedOrientedTriangles_2(triangs._UnderlyingPtr, repetitions), is_owning: true));
    }

    /// from local triangulations returns all oriented triangles with 3 or 2 repetitions each;
    /// if both outRep3 and outRep2 are necessary then it is faster to call this function than above one
    /// Generated from function `MR::findRepeatedOrientedTriangles`.
    public static unsafe void FindRepeatedOrientedTriangles(MR.Const_AllLocalTriangulations triangs, MR.Triangulation? outRep3, MR.Triangulation? outRep2)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findRepeatedOrientedTriangles_3", ExactSpelling = true)]
        extern static void __MR_findRepeatedOrientedTriangles_3(MR.Const_AllLocalTriangulations._Underlying *triangs, MR.Triangulation._Underlying *outRep3, MR.Triangulation._Underlying *outRep2);
        __MR_findRepeatedOrientedTriangles_3(triangs._UnderlyingPtr, outRep3 is not null ? outRep3._UnderlyingPtr : null, outRep2 is not null ? outRep2._UnderlyingPtr : null);
    }
}

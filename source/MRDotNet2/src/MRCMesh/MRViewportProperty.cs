public static partial class MR
{
    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::AffineXf3f>`.
    /// This is the const half of the class.
    public class Const_ViewportProperty_MRAffineXf3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ViewportProperty_MRAffineXf3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_AffineXf3f_Destroy(_Underlying *_this);
            __MR_ViewportProperty_MR_AffineXf3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportProperty_MRAffineXf3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportProperty_MRAffineXf3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRAffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_AffineXf3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::AffineXf3f>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRAffineXf3f(MR._ByValue_ViewportProperty_MRAffineXf3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRAffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_AffineXf3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::AffineXf3f>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRAffineXf3f(MR.Const_AffineXf3f def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRAffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_Construct(MR.Const_AffineXf3f._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_AffineXf3f_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::AffineXf3f>::ViewportProperty`.
        public static unsafe implicit operator Const_ViewportProperty_MRAffineXf3f(MR.Const_AffineXf3f def) {return new(def);}

        /// gets default property value
        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::get`.
        public unsafe MR.Const_AffineXf3f Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_get_const_0", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_get_const_0(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_AffineXf3f_get_const_0(_UnderlyingPtr), is_owning: false);
        }

        /// gets property value for given viewport: specific if available otherwise default one;
        /// \param isDef receives true if this viewport does not have specific value and default one is returned
        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::get`.
        public unsafe MR.Const_AffineXf3f Get(MR.ViewportId id, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_get_const_2", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_get_const_2(_Underlying *_this, MR.ViewportId id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ViewportProperty_MR_AffineXf3f_get_const_2(_UnderlyingPtr, id, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::AffineXf3f>`.
    /// This is the non-const half of the class.
    public class ViewportProperty_MRAffineXf3f : Const_ViewportProperty_MRAffineXf3f
    {
        internal unsafe ViewportProperty_MRAffineXf3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportProperty_MRAffineXf3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRAffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_AffineXf3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::AffineXf3f>::ViewportProperty`.
        public unsafe ViewportProperty_MRAffineXf3f(MR._ByValue_ViewportProperty_MRAffineXf3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRAffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_AffineXf3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::AffineXf3f>::ViewportProperty`.
        public unsafe ViewportProperty_MRAffineXf3f(MR.Const_AffineXf3f def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRAffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_Construct(MR.Const_AffineXf3f._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_AffineXf3f_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::AffineXf3f>::ViewportProperty`.
        public static unsafe implicit operator ViewportProperty_MRAffineXf3f(MR.Const_AffineXf3f def) {return new(def);}

        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::operator=`.
        public unsafe MR.ViewportProperty_MRAffineXf3f Assign(MR._ByValue_ViewportProperty_MRAffineXf3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRAffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *_other);
            return new(__MR_ViewportProperty_MR_AffineXf3f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// sets default property value
        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::set`.
        public unsafe void Set(MR.AffineXf3f def)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_set_1", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_AffineXf3f_set_1(_Underlying *_this, MR.AffineXf3f def);
            __MR_ViewportProperty_MR_AffineXf3f_set_1(_UnderlyingPtr, def);
        }

        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::get`.
        public unsafe new MR.Mut_AffineXf3f Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_get", ExactSpelling = true)]
            extern static MR.Mut_AffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_get(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_AffineXf3f_get(_UnderlyingPtr), is_owning: false);
        }

        /// returns direct access to value associated with given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::operator[]`.
        public unsafe MR.Mut_AffineXf3f Index(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_index", ExactSpelling = true)]
            extern static MR.Mut_AffineXf3f._Underlying *__MR_ViewportProperty_MR_AffineXf3f_index(_Underlying *_this, MR.ViewportId id);
            return new(__MR_ViewportProperty_MR_AffineXf3f_index(_UnderlyingPtr, id), is_owning: false);
        }

        /// sets specific property value for given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::set`.
        public unsafe void Set(MR.AffineXf3f v, MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_set_2", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_AffineXf3f_set_2(_Underlying *_this, MR.AffineXf3f v, MR.ViewportId id);
            __MR_ViewportProperty_MR_AffineXf3f_set_2(_UnderlyingPtr, v, id);
        }

        /// forgets specific property value for given viewport (or all viewports if !id);
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::reset`.
        public unsafe bool Reset(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_reset_1", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_AffineXf3f_reset_1(_Underlying *_this, MR.ViewportId id);
            return __MR_ViewportProperty_MR_AffineXf3f_reset_1(_UnderlyingPtr, id) != 0;
        }

        /// forgets specific property value for all viewports;
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::AffineXf3f>::reset`.
        public unsafe bool Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_AffineXf3f_reset_0", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_AffineXf3f_reset_0(_Underlying *_this);
            return __MR_ViewportProperty_MR_AffineXf3f_reset_0(_UnderlyingPtr) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `ViewportProperty_MRAffineXf3f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ViewportProperty_MRAffineXf3f`/`Const_ViewportProperty_MRAffineXf3f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ViewportProperty_MRAffineXf3f
    {
        internal readonly Const_ViewportProperty_MRAffineXf3f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ViewportProperty_MRAffineXf3f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ViewportProperty_MRAffineXf3f(Const_ViewportProperty_MRAffineXf3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ViewportProperty_MRAffineXf3f(Const_ViewportProperty_MRAffineXf3f arg) {return new(arg);}
        public _ByValue_ViewportProperty_MRAffineXf3f(MR.Misc._Moved<ViewportProperty_MRAffineXf3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ViewportProperty_MRAffineXf3f(MR.Misc._Moved<ViewportProperty_MRAffineXf3f> arg) {return new(arg);}

        /// Generated from constructor `MR::ViewportProperty<MR::AffineXf3f>::ViewportProperty`.
        public static unsafe implicit operator _ByValue_ViewportProperty_MRAffineXf3f(MR.Const_AffineXf3f def) {return new MR.ViewportProperty_MRAffineXf3f(def);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRAffineXf3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportProperty_MRAffineXf3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRAffineXf3f`/`Const_ViewportProperty_MRAffineXf3f` directly.
    public class _InOptMut_ViewportProperty_MRAffineXf3f
    {
        public ViewportProperty_MRAffineXf3f? Opt;

        public _InOptMut_ViewportProperty_MRAffineXf3f() {}
        public _InOptMut_ViewportProperty_MRAffineXf3f(ViewportProperty_MRAffineXf3f value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportProperty_MRAffineXf3f(ViewportProperty_MRAffineXf3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRAffineXf3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportProperty_MRAffineXf3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRAffineXf3f`/`Const_ViewportProperty_MRAffineXf3f` to pass it to the function.
    public class _InOptConst_ViewportProperty_MRAffineXf3f
    {
        public Const_ViewportProperty_MRAffineXf3f? Opt;

        public _InOptConst_ViewportProperty_MRAffineXf3f() {}
        public _InOptConst_ViewportProperty_MRAffineXf3f(Const_ViewportProperty_MRAffineXf3f value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportProperty_MRAffineXf3f(Const_ViewportProperty_MRAffineXf3f value) {return new(value);}

        /// Generated from constructor `MR::ViewportProperty<MR::AffineXf3f>::ViewportProperty`.
        public static unsafe implicit operator _InOptConst_ViewportProperty_MRAffineXf3f(MR.Const_AffineXf3f def) {return new MR.ViewportProperty_MRAffineXf3f(def);}
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::Color>`.
    /// This is the const half of the class.
    public class Const_ViewportProperty_MRColor : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ViewportProperty_MRColor(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Color_Destroy(_Underlying *_this);
            __MR_ViewportProperty_MR_Color_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportProperty_MRColor() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportProperty_MRColor() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRColor._Underlying *__MR_ViewportProperty_MR_Color_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_Color_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Color>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRColor(MR._ByValue_ViewportProperty_MRColor _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRColor._Underlying *__MR_ViewportProperty_MR_Color_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRColor._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Color_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Color>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRColor(MR.Const_Color def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRColor._Underlying *__MR_ViewportProperty_MR_Color_Construct(MR.Const_Color._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Color_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Color>::ViewportProperty`.
        public static unsafe implicit operator Const_ViewportProperty_MRColor(MR.Const_Color def) {return new(def);}

        /// gets default property value
        /// Generated from method `MR::ViewportProperty<MR::Color>::get`.
        public unsafe MR.Const_Color Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_get_const_0", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ViewportProperty_MR_Color_get_const_0(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_Color_get_const_0(_UnderlyingPtr), is_owning: false);
        }

        /// gets property value for given viewport: specific if available otherwise default one;
        /// \param isDef receives true if this viewport does not have specific value and default one is returned
        /// Generated from method `MR::ViewportProperty<MR::Color>::get`.
        public unsafe MR.Const_Color Get(MR.ViewportId id, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_get_const_2", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ViewportProperty_MR_Color_get_const_2(_Underlying *_this, MR.ViewportId id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ViewportProperty_MR_Color_get_const_2(_UnderlyingPtr, id, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::Color>`.
    /// This is the non-const half of the class.
    public class ViewportProperty_MRColor : Const_ViewportProperty_MRColor
    {
        internal unsafe ViewportProperty_MRColor(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportProperty_MRColor() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRColor._Underlying *__MR_ViewportProperty_MR_Color_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_Color_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Color>::ViewportProperty`.
        public unsafe ViewportProperty_MRColor(MR._ByValue_ViewportProperty_MRColor _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRColor._Underlying *__MR_ViewportProperty_MR_Color_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRColor._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Color_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Color>::ViewportProperty`.
        public unsafe ViewportProperty_MRColor(MR.Const_Color def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRColor._Underlying *__MR_ViewportProperty_MR_Color_Construct(MR.Const_Color._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Color_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Color>::ViewportProperty`.
        public static unsafe implicit operator ViewportProperty_MRColor(MR.Const_Color def) {return new(def);}

        /// Generated from method `MR::ViewportProperty<MR::Color>::operator=`.
        public unsafe MR.ViewportProperty_MRColor Assign(MR._ByValue_ViewportProperty_MRColor _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRColor._Underlying *__MR_ViewportProperty_MR_Color_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRColor._Underlying *_other);
            return new(__MR_ViewportProperty_MR_Color_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// sets default property value
        /// Generated from method `MR::ViewportProperty<MR::Color>::set`.
        public unsafe void Set(MR.Color def)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_set_1", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Color_set_1(_Underlying *_this, MR.Color def);
            __MR_ViewportProperty_MR_Color_set_1(_UnderlyingPtr, def);
        }

        /// Generated from method `MR::ViewportProperty<MR::Color>::get`.
        public unsafe new MR.Mut_Color Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_get", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_ViewportProperty_MR_Color_get(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_Color_get(_UnderlyingPtr), is_owning: false);
        }

        /// returns direct access to value associated with given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::Color>::operator[]`.
        public unsafe MR.Mut_Color Index(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_index", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_ViewportProperty_MR_Color_index(_Underlying *_this, MR.ViewportId id);
            return new(__MR_ViewportProperty_MR_Color_index(_UnderlyingPtr, id), is_owning: false);
        }

        /// sets specific property value for given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::Color>::set`.
        public unsafe void Set(MR.Color v, MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_set_2", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Color_set_2(_Underlying *_this, MR.Color v, MR.ViewportId id);
            __MR_ViewportProperty_MR_Color_set_2(_UnderlyingPtr, v, id);
        }

        /// forgets specific property value for given viewport (or all viewports if !id);
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::Color>::reset`.
        public unsafe bool Reset(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_reset_1", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_Color_reset_1(_Underlying *_this, MR.ViewportId id);
            return __MR_ViewportProperty_MR_Color_reset_1(_UnderlyingPtr, id) != 0;
        }

        /// forgets specific property value for all viewports;
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::Color>::reset`.
        public unsafe bool Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Color_reset_0", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_Color_reset_0(_Underlying *_this);
            return __MR_ViewportProperty_MR_Color_reset_0(_UnderlyingPtr) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `ViewportProperty_MRColor` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ViewportProperty_MRColor`/`Const_ViewportProperty_MRColor` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ViewportProperty_MRColor
    {
        internal readonly Const_ViewportProperty_MRColor? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ViewportProperty_MRColor() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ViewportProperty_MRColor(Const_ViewportProperty_MRColor new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ViewportProperty_MRColor(Const_ViewportProperty_MRColor arg) {return new(arg);}
        public _ByValue_ViewportProperty_MRColor(MR.Misc._Moved<ViewportProperty_MRColor> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ViewportProperty_MRColor(MR.Misc._Moved<ViewportProperty_MRColor> arg) {return new(arg);}

        /// Generated from constructor `MR::ViewportProperty<MR::Color>::ViewportProperty`.
        public static unsafe implicit operator _ByValue_ViewportProperty_MRColor(MR.Const_Color def) {return new MR.ViewportProperty_MRColor(def);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRColor` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportProperty_MRColor`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRColor`/`Const_ViewportProperty_MRColor` directly.
    public class _InOptMut_ViewportProperty_MRColor
    {
        public ViewportProperty_MRColor? Opt;

        public _InOptMut_ViewportProperty_MRColor() {}
        public _InOptMut_ViewportProperty_MRColor(ViewportProperty_MRColor value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportProperty_MRColor(ViewportProperty_MRColor value) {return new(value);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRColor` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportProperty_MRColor`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRColor`/`Const_ViewportProperty_MRColor` to pass it to the function.
    public class _InOptConst_ViewportProperty_MRColor
    {
        public Const_ViewportProperty_MRColor? Opt;

        public _InOptConst_ViewportProperty_MRColor() {}
        public _InOptConst_ViewportProperty_MRColor(Const_ViewportProperty_MRColor value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportProperty_MRColor(Const_ViewportProperty_MRColor value) {return new(value);}

        /// Generated from constructor `MR::ViewportProperty<MR::Color>::ViewportProperty`.
        public static unsafe implicit operator _InOptConst_ViewportProperty_MRColor(MR.Const_Color def) {return new MR.ViewportProperty_MRColor(def);}
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<unsigned char>`.
    /// This is the const half of the class.
    public class Const_ViewportProperty_UnsignedChar : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ViewportProperty_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_unsigned_char_Destroy(_Underlying *_this);
            __MR_ViewportProperty_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportProperty_UnsignedChar() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportProperty_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_UnsignedChar._Underlying *__MR_ViewportProperty_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<unsigned char>::ViewportProperty`.
        public unsafe Const_ViewportProperty_UnsignedChar(MR._ByValue_ViewportProperty_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_UnsignedChar._Underlying *__MR_ViewportProperty_unsigned_char_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_unsigned_char_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<unsigned char>::ViewportProperty`.
        public unsafe Const_ViewportProperty_UnsignedChar(byte def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_UnsignedChar._Underlying *__MR_ViewportProperty_unsigned_char_Construct(byte *def);
            _UnderlyingPtr = __MR_ViewportProperty_unsigned_char_Construct(&def);
        }

        /// Generated from constructor `MR::ViewportProperty<unsigned char>::ViewportProperty`.
        public static unsafe implicit operator Const_ViewportProperty_UnsignedChar(byte def) {return new(def);}

        /// gets default property value
        /// Generated from method `MR::ViewportProperty<unsigned char>::get`.
        public unsafe byte Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_get_const_0", ExactSpelling = true)]
            extern static byte *__MR_ViewportProperty_unsigned_char_get_const_0(_Underlying *_this);
            return *__MR_ViewportProperty_unsigned_char_get_const_0(_UnderlyingPtr);
        }

        /// gets property value for given viewport: specific if available otherwise default one;
        /// \param isDef receives true if this viewport does not have specific value and default one is returned
        /// Generated from method `MR::ViewportProperty<unsigned char>::get`.
        public unsafe byte Get(MR.ViewportId id, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_get_const_2", ExactSpelling = true)]
            extern static byte *__MR_ViewportProperty_unsigned_char_get_const_2(_Underlying *_this, MR.ViewportId id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ViewportProperty_unsigned_char_get_const_2(_UnderlyingPtr, id, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return *__ret;
        }
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<unsigned char>`.
    /// This is the non-const half of the class.
    public class ViewportProperty_UnsignedChar : Const_ViewportProperty_UnsignedChar
    {
        internal unsafe ViewportProperty_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportProperty_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_UnsignedChar._Underlying *__MR_ViewportProperty_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<unsigned char>::ViewportProperty`.
        public unsafe ViewportProperty_UnsignedChar(MR._ByValue_ViewportProperty_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_UnsignedChar._Underlying *__MR_ViewportProperty_unsigned_char_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_unsigned_char_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<unsigned char>::ViewportProperty`.
        public unsafe ViewportProperty_UnsignedChar(byte def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_UnsignedChar._Underlying *__MR_ViewportProperty_unsigned_char_Construct(byte *def);
            _UnderlyingPtr = __MR_ViewportProperty_unsigned_char_Construct(&def);
        }

        /// Generated from constructor `MR::ViewportProperty<unsigned char>::ViewportProperty`.
        public static unsafe implicit operator ViewportProperty_UnsignedChar(byte def) {return new(def);}

        /// Generated from method `MR::ViewportProperty<unsigned char>::operator=`.
        public unsafe MR.ViewportProperty_UnsignedChar Assign(MR._ByValue_ViewportProperty_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_UnsignedChar._Underlying *__MR_ViewportProperty_unsigned_char_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *_other);
            return new(__MR_ViewportProperty_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// sets default property value
        /// Generated from method `MR::ViewportProperty<unsigned char>::set`.
        public unsafe void Set(byte def)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_set_1", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_unsigned_char_set_1(_Underlying *_this, byte def);
            __MR_ViewportProperty_unsigned_char_set_1(_UnderlyingPtr, def);
        }

        /// Generated from method `MR::ViewportProperty<unsigned char>::get`.
        public unsafe new ref byte Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_get", ExactSpelling = true)]
            extern static byte *__MR_ViewportProperty_unsigned_char_get(_Underlying *_this);
            return ref *__MR_ViewportProperty_unsigned_char_get(_UnderlyingPtr);
        }

        /// returns direct access to value associated with given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<unsigned char>::operator[]`.
        public unsafe ref byte Index(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_index", ExactSpelling = true)]
            extern static byte *__MR_ViewportProperty_unsigned_char_index(_Underlying *_this, MR.ViewportId id);
            return ref *__MR_ViewportProperty_unsigned_char_index(_UnderlyingPtr, id);
        }

        /// sets specific property value for given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<unsigned char>::set`.
        public unsafe void Set(byte v, MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_set_2", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_unsigned_char_set_2(_Underlying *_this, byte v, MR.ViewportId id);
            __MR_ViewportProperty_unsigned_char_set_2(_UnderlyingPtr, v, id);
        }

        /// forgets specific property value for given viewport (or all viewports if !id);
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<unsigned char>::reset`.
        public unsafe bool Reset(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_reset_1", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_unsigned_char_reset_1(_Underlying *_this, MR.ViewportId id);
            return __MR_ViewportProperty_unsigned_char_reset_1(_UnderlyingPtr, id) != 0;
        }

        /// forgets specific property value for all viewports;
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<unsigned char>::reset`.
        public unsafe bool Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_unsigned_char_reset_0", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_unsigned_char_reset_0(_Underlying *_this);
            return __MR_ViewportProperty_unsigned_char_reset_0(_UnderlyingPtr) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `ViewportProperty_UnsignedChar` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ViewportProperty_UnsignedChar`/`Const_ViewportProperty_UnsignedChar` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ViewportProperty_UnsignedChar
    {
        internal readonly Const_ViewportProperty_UnsignedChar? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ViewportProperty_UnsignedChar() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ViewportProperty_UnsignedChar(Const_ViewportProperty_UnsignedChar new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ViewportProperty_UnsignedChar(Const_ViewportProperty_UnsignedChar arg) {return new(arg);}
        public _ByValue_ViewportProperty_UnsignedChar(MR.Misc._Moved<ViewportProperty_UnsignedChar> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ViewportProperty_UnsignedChar(MR.Misc._Moved<ViewportProperty_UnsignedChar> arg) {return new(arg);}

        /// Generated from constructor `MR::ViewportProperty<unsigned char>::ViewportProperty`.
        public static unsafe implicit operator _ByValue_ViewportProperty_UnsignedChar(byte def) {return new MR.ViewportProperty_UnsignedChar(def);}
    }

    /// This is used for optional parameters of class `ViewportProperty_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportProperty_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_UnsignedChar`/`Const_ViewportProperty_UnsignedChar` directly.
    public class _InOptMut_ViewportProperty_UnsignedChar
    {
        public ViewportProperty_UnsignedChar? Opt;

        public _InOptMut_ViewportProperty_UnsignedChar() {}
        public _InOptMut_ViewportProperty_UnsignedChar(ViewportProperty_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportProperty_UnsignedChar(ViewportProperty_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `ViewportProperty_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportProperty_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_UnsignedChar`/`Const_ViewportProperty_UnsignedChar` to pass it to the function.
    public class _InOptConst_ViewportProperty_UnsignedChar
    {
        public Const_ViewportProperty_UnsignedChar? Opt;

        public _InOptConst_ViewportProperty_UnsignedChar() {}
        public _InOptConst_ViewportProperty_UnsignedChar(Const_ViewportProperty_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportProperty_UnsignedChar(Const_ViewportProperty_UnsignedChar value) {return new(value);}

        /// Generated from constructor `MR::ViewportProperty<unsigned char>::ViewportProperty`.
        public static unsafe implicit operator _InOptConst_ViewportProperty_UnsignedChar(byte def) {return new MR.ViewportProperty_UnsignedChar(def);}
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>`.
    /// This is the const half of the class.
    public class Const_ViewportProperty_MRXfBasedCacheMRBox3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ViewportProperty_MRXfBasedCacheMRBox3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Destroy(_Underlying *_this);
            __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportProperty_MRXfBasedCacheMRBox3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportProperty_MRXfBasedCacheMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRXfBasedCacheMRBox3f(MR._ByValue_ViewportProperty_MRXfBasedCacheMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRXfBasedCacheMRBox3f(MR.Const_XfBasedCache_MRBox3f def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Construct(MR.Const_XfBasedCache_MRBox3f._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::ViewportProperty`.
        public static unsafe implicit operator Const_ViewportProperty_MRXfBasedCacheMRBox3f(MR.Const_XfBasedCache_MRBox3f def) {return new(def);}

        /// gets default property value
        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::get`.
        public unsafe MR.Const_XfBasedCache_MRBox3f Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get_const_0", ExactSpelling = true)]
            extern static MR.Const_XfBasedCache_MRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get_const_0(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get_const_0(_UnderlyingPtr), is_owning: false);
        }

        /// gets property value for given viewport: specific if available otherwise default one;
        /// \param isDef receives true if this viewport does not have specific value and default one is returned
        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::get`.
        public unsafe MR.Const_XfBasedCache_MRBox3f Get(MR.ViewportId id, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get_const_2", ExactSpelling = true)]
            extern static MR.Const_XfBasedCache_MRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get_const_2(_Underlying *_this, MR.ViewportId id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get_const_2(_UnderlyingPtr, id, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>`.
    /// This is the non-const half of the class.
    public class ViewportProperty_MRXfBasedCacheMRBox3f : Const_ViewportProperty_MRXfBasedCacheMRBox3f
    {
        internal unsafe ViewportProperty_MRXfBasedCacheMRBox3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportProperty_MRXfBasedCacheMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::ViewportProperty`.
        public unsafe ViewportProperty_MRXfBasedCacheMRBox3f(MR._ByValue_ViewportProperty_MRXfBasedCacheMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::ViewportProperty`.
        public unsafe ViewportProperty_MRXfBasedCacheMRBox3f(MR.Const_XfBasedCache_MRBox3f def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Construct(MR.Const_XfBasedCache_MRBox3f._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::ViewportProperty`.
        public static unsafe implicit operator ViewportProperty_MRXfBasedCacheMRBox3f(MR.Const_XfBasedCache_MRBox3f def) {return new(def);}

        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::operator=`.
        public unsafe MR.ViewportProperty_MRXfBasedCacheMRBox3f Assign(MR._ByValue_ViewportProperty_MRXfBasedCacheMRBox3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRXfBasedCacheMRBox3f._Underlying *_other);
            return new(__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// sets default property value
        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::set`.
        public unsafe void Set(MR.Const_XfBasedCache_MRBox3f def)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_set_1", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_set_1(_Underlying *_this, MR.XfBasedCache_MRBox3f._Underlying *def);
            __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_set_1(_UnderlyingPtr, def._UnderlyingPtr);
        }

        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::get`.
        public unsafe new MR.XfBasedCache_MRBox3f Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get", ExactSpelling = true)]
            extern static MR.XfBasedCache_MRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_get(_UnderlyingPtr), is_owning: false);
        }

        /// returns direct access to value associated with given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::operator[]`.
        public unsafe MR.XfBasedCache_MRBox3f Index(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_index", ExactSpelling = true)]
            extern static MR.XfBasedCache_MRBox3f._Underlying *__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_index(_Underlying *_this, MR.ViewportId id);
            return new(__MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_index(_UnderlyingPtr, id), is_owning: false);
        }

        /// sets specific property value for given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::set`.
        public unsafe void Set(MR.Const_XfBasedCache_MRBox3f v, MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_set_2", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_set_2(_Underlying *_this, MR.XfBasedCache_MRBox3f._Underlying *v, MR.ViewportId id);
            __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_set_2(_UnderlyingPtr, v._UnderlyingPtr, id);
        }

        /// forgets specific property value for given viewport (or all viewports if !id);
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::reset`.
        public unsafe bool Reset(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_reset_1", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_reset_1(_Underlying *_this, MR.ViewportId id);
            return __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_reset_1(_UnderlyingPtr, id) != 0;
        }

        /// forgets specific property value for all viewports;
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::reset`.
        public unsafe bool Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_reset_0", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_reset_0(_Underlying *_this);
            return __MR_ViewportProperty_MR_XfBasedCache_MR_Box3f_reset_0(_UnderlyingPtr) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `ViewportProperty_MRXfBasedCacheMRBox3f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ViewportProperty_MRXfBasedCacheMRBox3f`/`Const_ViewportProperty_MRXfBasedCacheMRBox3f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ViewportProperty_MRXfBasedCacheMRBox3f
    {
        internal readonly Const_ViewportProperty_MRXfBasedCacheMRBox3f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ViewportProperty_MRXfBasedCacheMRBox3f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ViewportProperty_MRXfBasedCacheMRBox3f(Const_ViewportProperty_MRXfBasedCacheMRBox3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ViewportProperty_MRXfBasedCacheMRBox3f(Const_ViewportProperty_MRXfBasedCacheMRBox3f arg) {return new(arg);}
        public _ByValue_ViewportProperty_MRXfBasedCacheMRBox3f(MR.Misc._Moved<ViewportProperty_MRXfBasedCacheMRBox3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ViewportProperty_MRXfBasedCacheMRBox3f(MR.Misc._Moved<ViewportProperty_MRXfBasedCacheMRBox3f> arg) {return new(arg);}

        /// Generated from constructor `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::ViewportProperty`.
        public static unsafe implicit operator _ByValue_ViewportProperty_MRXfBasedCacheMRBox3f(MR.Const_XfBasedCache_MRBox3f def) {return new MR.ViewportProperty_MRXfBasedCacheMRBox3f(def);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRXfBasedCacheMRBox3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportProperty_MRXfBasedCacheMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRXfBasedCacheMRBox3f`/`Const_ViewportProperty_MRXfBasedCacheMRBox3f` directly.
    public class _InOptMut_ViewportProperty_MRXfBasedCacheMRBox3f
    {
        public ViewportProperty_MRXfBasedCacheMRBox3f? Opt;

        public _InOptMut_ViewportProperty_MRXfBasedCacheMRBox3f() {}
        public _InOptMut_ViewportProperty_MRXfBasedCacheMRBox3f(ViewportProperty_MRXfBasedCacheMRBox3f value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportProperty_MRXfBasedCacheMRBox3f(ViewportProperty_MRXfBasedCacheMRBox3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRXfBasedCacheMRBox3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportProperty_MRXfBasedCacheMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRXfBasedCacheMRBox3f`/`Const_ViewportProperty_MRXfBasedCacheMRBox3f` to pass it to the function.
    public class _InOptConst_ViewportProperty_MRXfBasedCacheMRBox3f
    {
        public Const_ViewportProperty_MRXfBasedCacheMRBox3f? Opt;

        public _InOptConst_ViewportProperty_MRXfBasedCacheMRBox3f() {}
        public _InOptConst_ViewportProperty_MRXfBasedCacheMRBox3f(Const_ViewportProperty_MRXfBasedCacheMRBox3f value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportProperty_MRXfBasedCacheMRBox3f(Const_ViewportProperty_MRXfBasedCacheMRBox3f value) {return new(value);}

        /// Generated from constructor `MR::ViewportProperty<MR::XfBasedCache<MR::Box3f>>::ViewportProperty`.
        public static unsafe implicit operator _InOptConst_ViewportProperty_MRXfBasedCacheMRBox3f(MR.Const_XfBasedCache_MRBox3f def) {return new MR.ViewportProperty_MRXfBasedCacheMRBox3f(def);}
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::Vector4<unsigned char>>`.
    /// This is the const half of the class.
    public class Const_ViewportProperty_MRVector4UnsignedChar : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ViewportProperty_MRVector4UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Vector4_unsigned_char_Destroy(_Underlying *_this);
            __MR_ViewportProperty_MR_Vector4_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportProperty_MRVector4UnsignedChar() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportProperty_MRVector4UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRVector4UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_Vector4_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Vector4<unsigned char>>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRVector4UnsignedChar(MR._ByValue_ViewportProperty_MRVector4UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRVector4UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRVector4UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Vector4_unsigned_char_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Vector4<unsigned char>>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRVector4UnsignedChar(MR.Const_Vector4_UnsignedChar def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRVector4UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_Construct(MR.Const_Vector4_UnsignedChar._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Vector4_unsigned_char_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Vector4<unsigned char>>::ViewportProperty`.
        public static unsafe implicit operator Const_ViewportProperty_MRVector4UnsignedChar(MR.Const_Vector4_UnsignedChar def) {return new(def);}

        /// gets default property value
        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::get`.
        public unsafe MR.Const_Vector4_UnsignedChar Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_get_const_0", ExactSpelling = true)]
            extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_get_const_0(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_Vector4_unsigned_char_get_const_0(_UnderlyingPtr), is_owning: false);
        }

        /// gets property value for given viewport: specific if available otherwise default one;
        /// \param isDef receives true if this viewport does not have specific value and default one is returned
        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::get`.
        public unsafe MR.Const_Vector4_UnsignedChar Get(MR.ViewportId id, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_get_const_2", ExactSpelling = true)]
            extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_get_const_2(_Underlying *_this, MR.ViewportId id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ViewportProperty_MR_Vector4_unsigned_char_get_const_2(_UnderlyingPtr, id, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::Vector4<unsigned char>>`.
    /// This is the non-const half of the class.
    public class ViewportProperty_MRVector4UnsignedChar : Const_ViewportProperty_MRVector4UnsignedChar
    {
        internal unsafe ViewportProperty_MRVector4UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportProperty_MRVector4UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRVector4UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_Vector4_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Vector4<unsigned char>>::ViewportProperty`.
        public unsafe ViewportProperty_MRVector4UnsignedChar(MR._ByValue_ViewportProperty_MRVector4UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRVector4UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRVector4UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Vector4_unsigned_char_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Vector4<unsigned char>>::ViewportProperty`.
        public unsafe ViewportProperty_MRVector4UnsignedChar(MR.Const_Vector4_UnsignedChar def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRVector4UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_Construct(MR.Const_Vector4_UnsignedChar._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Vector4_unsigned_char_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Vector4<unsigned char>>::ViewportProperty`.
        public static unsafe implicit operator ViewportProperty_MRVector4UnsignedChar(MR.Const_Vector4_UnsignedChar def) {return new(def);}

        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::operator=`.
        public unsafe MR.ViewportProperty_MRVector4UnsignedChar Assign(MR._ByValue_ViewportProperty_MRVector4UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRVector4UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRVector4UnsignedChar._Underlying *_other);
            return new(__MR_ViewportProperty_MR_Vector4_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// sets default property value
        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::set`.
        public unsafe void Set(MR.Const_Vector4_UnsignedChar def)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_set_1", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Vector4_unsigned_char_set_1(_Underlying *_this, MR.Vector4_UnsignedChar._Underlying *def);
            __MR_ViewportProperty_MR_Vector4_unsigned_char_set_1(_UnderlyingPtr, def._UnderlyingPtr);
        }

        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::get`.
        public unsafe new MR.Vector4_UnsignedChar Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_get", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_get(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_Vector4_unsigned_char_get(_UnderlyingPtr), is_owning: false);
        }

        /// returns direct access to value associated with given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::operator[]`.
        public unsafe MR.Vector4_UnsignedChar Index(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_index", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_ViewportProperty_MR_Vector4_unsigned_char_index(_Underlying *_this, MR.ViewportId id);
            return new(__MR_ViewportProperty_MR_Vector4_unsigned_char_index(_UnderlyingPtr, id), is_owning: false);
        }

        /// sets specific property value for given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::set`.
        public unsafe void Set(MR.Const_Vector4_UnsignedChar v, MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_set_2", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Vector4_unsigned_char_set_2(_Underlying *_this, MR.Vector4_UnsignedChar._Underlying *v, MR.ViewportId id);
            __MR_ViewportProperty_MR_Vector4_unsigned_char_set_2(_UnderlyingPtr, v._UnderlyingPtr, id);
        }

        /// forgets specific property value for given viewport (or all viewports if !id);
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::reset`.
        public unsafe bool Reset(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_reset_1", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_Vector4_unsigned_char_reset_1(_Underlying *_this, MR.ViewportId id);
            return __MR_ViewportProperty_MR_Vector4_unsigned_char_reset_1(_UnderlyingPtr, id) != 0;
        }

        /// forgets specific property value for all viewports;
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::Vector4<unsigned char>>::reset`.
        public unsafe bool Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Vector4_unsigned_char_reset_0", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_Vector4_unsigned_char_reset_0(_Underlying *_this);
            return __MR_ViewportProperty_MR_Vector4_unsigned_char_reset_0(_UnderlyingPtr) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `ViewportProperty_MRVector4UnsignedChar` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ViewportProperty_MRVector4UnsignedChar`/`Const_ViewportProperty_MRVector4UnsignedChar` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ViewportProperty_MRVector4UnsignedChar
    {
        internal readonly Const_ViewportProperty_MRVector4UnsignedChar? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ViewportProperty_MRVector4UnsignedChar() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ViewportProperty_MRVector4UnsignedChar(Const_ViewportProperty_MRVector4UnsignedChar new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ViewportProperty_MRVector4UnsignedChar(Const_ViewportProperty_MRVector4UnsignedChar arg) {return new(arg);}
        public _ByValue_ViewportProperty_MRVector4UnsignedChar(MR.Misc._Moved<ViewportProperty_MRVector4UnsignedChar> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ViewportProperty_MRVector4UnsignedChar(MR.Misc._Moved<ViewportProperty_MRVector4UnsignedChar> arg) {return new(arg);}

        /// Generated from constructor `MR::ViewportProperty<MR::Vector4<unsigned char>>::ViewportProperty`.
        public static unsafe implicit operator _ByValue_ViewportProperty_MRVector4UnsignedChar(MR.Const_Vector4_UnsignedChar def) {return new MR.ViewportProperty_MRVector4UnsignedChar(def);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRVector4UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportProperty_MRVector4UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRVector4UnsignedChar`/`Const_ViewportProperty_MRVector4UnsignedChar` directly.
    public class _InOptMut_ViewportProperty_MRVector4UnsignedChar
    {
        public ViewportProperty_MRVector4UnsignedChar? Opt;

        public _InOptMut_ViewportProperty_MRVector4UnsignedChar() {}
        public _InOptMut_ViewportProperty_MRVector4UnsignedChar(ViewportProperty_MRVector4UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportProperty_MRVector4UnsignedChar(ViewportProperty_MRVector4UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRVector4UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportProperty_MRVector4UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRVector4UnsignedChar`/`Const_ViewportProperty_MRVector4UnsignedChar` to pass it to the function.
    public class _InOptConst_ViewportProperty_MRVector4UnsignedChar
    {
        public Const_ViewportProperty_MRVector4UnsignedChar? Opt;

        public _InOptConst_ViewportProperty_MRVector4UnsignedChar() {}
        public _InOptConst_ViewportProperty_MRVector4UnsignedChar(Const_ViewportProperty_MRVector4UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportProperty_MRVector4UnsignedChar(Const_ViewportProperty_MRVector4UnsignedChar value) {return new(value);}

        /// Generated from constructor `MR::ViewportProperty<MR::Vector4<unsigned char>>::ViewportProperty`.
        public static unsafe implicit operator _InOptConst_ViewportProperty_MRVector4UnsignedChar(MR.Const_Vector4_UnsignedChar def) {return new MR.ViewportProperty_MRVector4UnsignedChar(def);}
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::Matrix3f>`.
    /// This is the const half of the class.
    public class Const_ViewportProperty_MRMatrix3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ViewportProperty_MRMatrix3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Matrix3f_Destroy(_Underlying *_this);
            __MR_ViewportProperty_MR_Matrix3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportProperty_MRMatrix3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportProperty_MRMatrix3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRMatrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_Matrix3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Matrix3f>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRMatrix3f(MR._ByValue_ViewportProperty_MRMatrix3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRMatrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRMatrix3f._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Matrix3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Matrix3f>::ViewportProperty`.
        public unsafe Const_ViewportProperty_MRMatrix3f(MR.Const_Matrix3f def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRMatrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_Construct(MR.Const_Matrix3f._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Matrix3f_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Matrix3f>::ViewportProperty`.
        public static unsafe implicit operator Const_ViewportProperty_MRMatrix3f(MR.Const_Matrix3f def) {return new(def);}

        /// gets default property value
        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::get`.
        public unsafe MR.Const_Matrix3f Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_get_const_0", ExactSpelling = true)]
            extern static MR.Const_Matrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_get_const_0(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_Matrix3f_get_const_0(_UnderlyingPtr), is_owning: false);
        }

        /// gets property value for given viewport: specific if available otherwise default one;
        /// \param isDef receives true if this viewport does not have specific value and default one is returned
        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::get`.
        public unsafe MR.Const_Matrix3f Get(MR.ViewportId id, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_get_const_2", ExactSpelling = true)]
            extern static MR.Const_Matrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_get_const_2(_Underlying *_this, MR.ViewportId id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ViewportProperty_MR_Matrix3f_get_const_2(_UnderlyingPtr, id, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }
    }

    /// storage of some viewport-dependent property,
    /// which has some default value for all viewports and special values for some specific viewports
    /// Generated from class `MR::ViewportProperty<MR::Matrix3f>`.
    /// This is the non-const half of the class.
    public class ViewportProperty_MRMatrix3f : Const_ViewportProperty_MRMatrix3f
    {
        internal unsafe ViewportProperty_MRMatrix3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportProperty_MRMatrix3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRMatrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportProperty_MR_Matrix3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Matrix3f>::ViewportProperty`.
        public unsafe ViewportProperty_MRMatrix3f(MR._ByValue_ViewportProperty_MRMatrix3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRMatrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRMatrix3f._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Matrix3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Matrix3f>::ViewportProperty`.
        public unsafe ViewportProperty_MRMatrix3f(MR.Const_Matrix3f def) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_Construct", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRMatrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_Construct(MR.Const_Matrix3f._Underlying *def);
            _UnderlyingPtr = __MR_ViewportProperty_MR_Matrix3f_Construct(def._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportProperty<MR::Matrix3f>::ViewportProperty`.
        public static unsafe implicit operator ViewportProperty_MRMatrix3f(MR.Const_Matrix3f def) {return new(def);}

        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::operator=`.
        public unsafe MR.ViewportProperty_MRMatrix3f Assign(MR._ByValue_ViewportProperty_MRMatrix3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ViewportProperty_MRMatrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ViewportProperty_MRMatrix3f._Underlying *_other);
            return new(__MR_ViewportProperty_MR_Matrix3f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// sets default property value
        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::set`.
        public unsafe void Set(MR.Matrix3f def)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_set_1", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Matrix3f_set_1(_Underlying *_this, MR.Matrix3f def);
            __MR_ViewportProperty_MR_Matrix3f_set_1(_UnderlyingPtr, def);
        }

        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::get`.
        public unsafe new MR.Mut_Matrix3f Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_get", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_get(_Underlying *_this);
            return new(__MR_ViewportProperty_MR_Matrix3f_get(_UnderlyingPtr), is_owning: false);
        }

        /// returns direct access to value associated with given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::operator[]`.
        public unsafe MR.Mut_Matrix3f Index(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_index", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_ViewportProperty_MR_Matrix3f_index(_Underlying *_this, MR.ViewportId id);
            return new(__MR_ViewportProperty_MR_Matrix3f_index(_UnderlyingPtr, id), is_owning: false);
        }

        /// sets specific property value for given viewport (or default value if !id)
        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::set`.
        public unsafe void Set(MR.Matrix3f v, MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_set_2", ExactSpelling = true)]
            extern static void __MR_ViewportProperty_MR_Matrix3f_set_2(_Underlying *_this, MR.Matrix3f v, MR.ViewportId id);
            __MR_ViewportProperty_MR_Matrix3f_set_2(_UnderlyingPtr, v, id);
        }

        /// forgets specific property value for given viewport (or all viewports if !id);
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::reset`.
        public unsafe bool Reset(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_reset_1", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_Matrix3f_reset_1(_Underlying *_this, MR.ViewportId id);
            return __MR_ViewportProperty_MR_Matrix3f_reset_1(_UnderlyingPtr, id) != 0;
        }

        /// forgets specific property value for all viewports;
        /// returns true if any specific value was removed
        /// Generated from method `MR::ViewportProperty<MR::Matrix3f>::reset`.
        public unsafe bool Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportProperty_MR_Matrix3f_reset_0", ExactSpelling = true)]
            extern static byte __MR_ViewportProperty_MR_Matrix3f_reset_0(_Underlying *_this);
            return __MR_ViewportProperty_MR_Matrix3f_reset_0(_UnderlyingPtr) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `ViewportProperty_MRMatrix3f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ViewportProperty_MRMatrix3f`/`Const_ViewportProperty_MRMatrix3f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ViewportProperty_MRMatrix3f
    {
        internal readonly Const_ViewportProperty_MRMatrix3f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ViewportProperty_MRMatrix3f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ViewportProperty_MRMatrix3f(Const_ViewportProperty_MRMatrix3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ViewportProperty_MRMatrix3f(Const_ViewportProperty_MRMatrix3f arg) {return new(arg);}
        public _ByValue_ViewportProperty_MRMatrix3f(MR.Misc._Moved<ViewportProperty_MRMatrix3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ViewportProperty_MRMatrix3f(MR.Misc._Moved<ViewportProperty_MRMatrix3f> arg) {return new(arg);}

        /// Generated from constructor `MR::ViewportProperty<MR::Matrix3f>::ViewportProperty`.
        public static unsafe implicit operator _ByValue_ViewportProperty_MRMatrix3f(MR.Const_Matrix3f def) {return new MR.ViewportProperty_MRMatrix3f(def);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRMatrix3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportProperty_MRMatrix3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRMatrix3f`/`Const_ViewportProperty_MRMatrix3f` directly.
    public class _InOptMut_ViewportProperty_MRMatrix3f
    {
        public ViewportProperty_MRMatrix3f? Opt;

        public _InOptMut_ViewportProperty_MRMatrix3f() {}
        public _InOptMut_ViewportProperty_MRMatrix3f(ViewportProperty_MRMatrix3f value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportProperty_MRMatrix3f(ViewportProperty_MRMatrix3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `ViewportProperty_MRMatrix3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportProperty_MRMatrix3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportProperty_MRMatrix3f`/`Const_ViewportProperty_MRMatrix3f` to pass it to the function.
    public class _InOptConst_ViewportProperty_MRMatrix3f
    {
        public Const_ViewportProperty_MRMatrix3f? Opt;

        public _InOptConst_ViewportProperty_MRMatrix3f() {}
        public _InOptConst_ViewportProperty_MRMatrix3f(Const_ViewportProperty_MRMatrix3f value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportProperty_MRMatrix3f(Const_ViewportProperty_MRMatrix3f value) {return new(value);}

        /// Generated from constructor `MR::ViewportProperty<MR::Matrix3f>::ViewportProperty`.
        public static unsafe implicit operator _InOptConst_ViewportProperty_MRMatrix3f(MR.Const_Matrix3f def) {return new MR.ViewportProperty_MRMatrix3f(def);}
    }
}

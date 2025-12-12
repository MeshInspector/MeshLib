public static partial class MR
{
    /**
    * \struct MR::DecimatePolylineSettings
    * \brief Parameters structure for MR::decimatePolyline
    *
    *
    * \sa \ref decimatePolyline
    */
    /// Generated from class `MR::DecimatePolylineSettings<MR::Vector2f>`.
    /// This is the const half of the class.
    public class Const_DecimatePolylineSettings_MRVector2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DecimatePolylineSettings_MRVector2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Destroy", ExactSpelling = true)]
            extern static void __MR_DecimatePolylineSettings_MR_Vector2f_Destroy(_Underlying *_this);
            __MR_DecimatePolylineSettings_MR_Vector2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DecimatePolylineSettings_MRVector2f() {Dispose(false);}

        /// Limit from above on the maximum distance from moved vertices to original contour
        public unsafe float MaxError
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_maxError", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector2f_Get_maxError(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector2f_Get_maxError(_UnderlyingPtr);
            }
        }

        /// Maximal possible edge length created during decimation
        public unsafe float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector2f_Get_maxEdgeLen(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector2f_Get_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Stabilizer is dimensionless coefficient.
        /// The larger is stabilizer, the more Decimator will strive to retain the density of input points.
        /// If stabilizer is zero, then only the shape of input line will be preserved.
        public unsafe float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_stabilizer", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector2f_Get_stabilizer(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector2f_Get_stabilizer(_UnderlyingPtr);
            }
        }

        /// if true then after each edge collapse the position of remaining vertex is optimized to
        /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
        public unsafe bool OptimizeVertexPos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_optimizeVertexPos", ExactSpelling = true)]
                extern static bool *__MR_DecimatePolylineSettings_MR_Vector2f_Get_optimizeVertexPos(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector2f_Get_optimizeVertexPos(_UnderlyingPtr);
            }
        }

        /// Limit on the number of deleted vertices
        public unsafe int MaxDeletedVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_maxDeletedVertices", ExactSpelling = true)]
                extern static int *__MR_DecimatePolylineSettings_MR_Vector2f_Get_maxDeletedVertices(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector2f_Get_maxDeletedVertices(_UnderlyingPtr);
            }
        }

        /// Region of the polyline to be decimated, it is updated during the operation
        /// Remain nullptr to include the whole polyline
        public unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_region", ExactSpelling = true)]
                extern static void **__MR_DecimatePolylineSettings_MR_Vector2f_Get_region(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_Get_region(_UnderlyingPtr);
            }
        }

        /// Whether to allow collapsing edges with at least one vertex on the end of not-closed polyline
        /// (or on region boundary if region is given);
        /// if touchBdVertices is false then boundary vertices are strictly fixed
        public unsafe bool TouchBdVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_touchBdVertices", ExactSpelling = true)]
                extern static bool *__MR_DecimatePolylineSettings_MR_Vector2f_Get_touchBdVertices(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector2f_Get_touchBdVertices(_UnderlyingPtr);
            }
        }

        /**
        * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
        * \details It receives the edge being collapsed: its destination vertex will disappear,
        * and its origin vertex will get new position (provided as the second argument) after collapse;
        * If the callback returns false, then the collapse is prohibited
        */
        public unsafe MR.Std.Const_Function_BoolFuncFromMREdgeIdConstMRVector2fRef PreCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_preCollapse", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromMREdgeIdConstMRVector2fRef._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_Get_preCollapse(_Underlying *_this);
                return new(__MR_DecimatePolylineSettings_MR_Vector2f_Get_preCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief The user can provide this optional callback for adjusting error introduced by this
        * edge collapse and the collapse position.
        * \details On input the callback gets the squared error and position computed by standard means,
        * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
        * This callback can be called from many threads in parallel and must be thread-safe.
        * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
        */
        public unsafe MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector2fRef AdjustCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_adjustCollapse", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector2fRef._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_Get_adjustCollapse(_Underlying *_this);
                return new(__MR_DecimatePolylineSettings_MR_Vector2f_Get_adjustCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief  If not null, then
        * on input: if the vector is not empty then it is taken for initialization instead of form computation for all vertices;
        * on output: quadratic form for each remaining vertex is returned there
        */
        public unsafe ref void * VertForms
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_Get_vertForms", ExactSpelling = true)]
                extern static void **__MR_DecimatePolylineSettings_MR_Vector2f_Get_vertForms(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_Get_vertForms(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DecimatePolylineSettings_MRVector2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector2f._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector2f_DefaultConstruct();
        }

        /// Constructs `MR::DecimatePolylineSettings<MR::Vector2f>` elementwise.
        public unsafe Const_DecimatePolylineSettings_MRVector2f(float maxError, float maxEdgeLen, float stabilizer, bool optimizeVertexPos, int maxDeletedVertices, MR.VertBitSet? region, bool touchBdVertices, MR.Std._ByValue_Function_BoolFuncFromMREdgeIdConstMRVector2fRef preCollapse, MR.Std._ByValue_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector2fRef adjustCollapse, MR.Vector_MRQuadraticForm2f_MRVertId? vertForms) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector2f._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_ConstructFrom(float maxError, float maxEdgeLen, float stabilizer, byte optimizeVertexPos, int maxDeletedVertices, MR.VertBitSet._Underlying *region, byte touchBdVertices, MR.Misc._PassBy preCollapse_pass_by, MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector2fRef._Underlying *preCollapse, MR.Misc._PassBy adjustCollapse_pass_by, MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector2fRef._Underlying *adjustCollapse, MR.Vector_MRQuadraticForm2f_MRVertId._Underlying *vertForms);
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector2f_ConstructFrom(maxError, maxEdgeLen, stabilizer, optimizeVertexPos ? (byte)1 : (byte)0, maxDeletedVertices, region is not null ? region._UnderlyingPtr : null, touchBdVertices ? (byte)1 : (byte)0, preCollapse.PassByMode, preCollapse.Value is not null ? preCollapse.Value._UnderlyingPtr : null, adjustCollapse.PassByMode, adjustCollapse.Value is not null ? adjustCollapse.Value._UnderlyingPtr : null, vertForms is not null ? vertForms._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DecimatePolylineSettings<MR::Vector2f>::DecimatePolylineSettings`.
        public unsafe Const_DecimatePolylineSettings_MRVector2f(MR._ByValue_DecimatePolylineSettings_MRVector2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector2f._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DecimatePolylineSettings_MRVector2f._Underlying *_other);
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector2f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /**
    * \struct MR::DecimatePolylineSettings
    * \brief Parameters structure for MR::decimatePolyline
    *
    *
    * \sa \ref decimatePolyline
    */
    /// Generated from class `MR::DecimatePolylineSettings<MR::Vector2f>`.
    /// This is the non-const half of the class.
    public class DecimatePolylineSettings_MRVector2f : Const_DecimatePolylineSettings_MRVector2f
    {
        internal unsafe DecimatePolylineSettings_MRVector2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Limit from above on the maximum distance from moved vertices to original contour
        public new unsafe ref float MaxError
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxError", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxError(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxError(_UnderlyingPtr);
            }
        }

        /// Maximal possible edge length created during decimation
        public new unsafe ref float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxEdgeLen(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Stabilizer is dimensionless coefficient.
        /// The larger is stabilizer, the more Decimator will strive to retain the density of input points.
        /// If stabilizer is zero, then only the shape of input line will be preserved.
        public new unsafe ref float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_stabilizer", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_stabilizer(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_stabilizer(_UnderlyingPtr);
            }
        }

        /// if true then after each edge collapse the position of remaining vertex is optimized to
        /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
        public new unsafe ref bool OptimizeVertexPos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_optimizeVertexPos", ExactSpelling = true)]
                extern static bool *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_optimizeVertexPos(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_optimizeVertexPos(_UnderlyingPtr);
            }
        }

        /// Limit on the number of deleted vertices
        public new unsafe ref int MaxDeletedVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxDeletedVertices", ExactSpelling = true)]
                extern static int *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxDeletedVertices(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_maxDeletedVertices(_UnderlyingPtr);
            }
        }

        /// Region of the polyline to be decimated, it is updated during the operation
        /// Remain nullptr to include the whole polyline
        public new unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_region(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Whether to allow collapsing edges with at least one vertex on the end of not-closed polyline
        /// (or on region boundary if region is given);
        /// if touchBdVertices is false then boundary vertices are strictly fixed
        public new unsafe ref bool TouchBdVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_touchBdVertices", ExactSpelling = true)]
                extern static bool *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_touchBdVertices(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_touchBdVertices(_UnderlyingPtr);
            }
        }

        /**
        * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
        * \details It receives the edge being collapsed: its destination vertex will disappear,
        * and its origin vertex will get new position (provided as the second argument) after collapse;
        * If the callback returns false, then the collapse is prohibited
        */
        public new unsafe MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector2fRef PreCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_preCollapse", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector2fRef._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_preCollapse(_Underlying *_this);
                return new(__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_preCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief The user can provide this optional callback for adjusting error introduced by this
        * edge collapse and the collapse position.
        * \details On input the callback gets the squared error and position computed by standard means,
        * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
        * This callback can be called from many threads in parallel and must be thread-safe.
        * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
        */
        public new unsafe MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector2fRef AdjustCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_adjustCollapse", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector2fRef._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_adjustCollapse(_Underlying *_this);
                return new(__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_adjustCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief  If not null, then
        * on input: if the vector is not empty then it is taken for initialization instead of form computation for all vertices;
        * on output: quadratic form for each remaining vertex is returned there
        */
        public new unsafe ref void * VertForms
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_vertForms", ExactSpelling = true)]
                extern static void **__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_vertForms(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector2f_GetMutable_vertForms(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DecimatePolylineSettings_MRVector2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector2f._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector2f_DefaultConstruct();
        }

        /// Constructs `MR::DecimatePolylineSettings<MR::Vector2f>` elementwise.
        public unsafe DecimatePolylineSettings_MRVector2f(float maxError, float maxEdgeLen, float stabilizer, bool optimizeVertexPos, int maxDeletedVertices, MR.VertBitSet? region, bool touchBdVertices, MR.Std._ByValue_Function_BoolFuncFromMREdgeIdConstMRVector2fRef preCollapse, MR.Std._ByValue_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector2fRef adjustCollapse, MR.Vector_MRQuadraticForm2f_MRVertId? vertForms) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector2f._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_ConstructFrom(float maxError, float maxEdgeLen, float stabilizer, byte optimizeVertexPos, int maxDeletedVertices, MR.VertBitSet._Underlying *region, byte touchBdVertices, MR.Misc._PassBy preCollapse_pass_by, MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector2fRef._Underlying *preCollapse, MR.Misc._PassBy adjustCollapse_pass_by, MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector2fRef._Underlying *adjustCollapse, MR.Vector_MRQuadraticForm2f_MRVertId._Underlying *vertForms);
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector2f_ConstructFrom(maxError, maxEdgeLen, stabilizer, optimizeVertexPos ? (byte)1 : (byte)0, maxDeletedVertices, region is not null ? region._UnderlyingPtr : null, touchBdVertices ? (byte)1 : (byte)0, preCollapse.PassByMode, preCollapse.Value is not null ? preCollapse.Value._UnderlyingPtr : null, adjustCollapse.PassByMode, adjustCollapse.Value is not null ? adjustCollapse.Value._UnderlyingPtr : null, vertForms is not null ? vertForms._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DecimatePolylineSettings<MR::Vector2f>::DecimatePolylineSettings`.
        public unsafe DecimatePolylineSettings_MRVector2f(MR._ByValue_DecimatePolylineSettings_MRVector2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector2f._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DecimatePolylineSettings_MRVector2f._Underlying *_other);
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector2f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DecimatePolylineSettings<MR::Vector2f>::operator=`.
        public unsafe MR.DecimatePolylineSettings_MRVector2f Assign(MR._ByValue_DecimatePolylineSettings_MRVector2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector2f._Underlying *__MR_DecimatePolylineSettings_MR_Vector2f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DecimatePolylineSettings_MRVector2f._Underlying *_other);
            return new(__MR_DecimatePolylineSettings_MR_Vector2f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DecimatePolylineSettings_MRVector2f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DecimatePolylineSettings_MRVector2f`/`Const_DecimatePolylineSettings_MRVector2f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DecimatePolylineSettings_MRVector2f
    {
        internal readonly Const_DecimatePolylineSettings_MRVector2f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DecimatePolylineSettings_MRVector2f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DecimatePolylineSettings_MRVector2f(Const_DecimatePolylineSettings_MRVector2f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DecimatePolylineSettings_MRVector2f(Const_DecimatePolylineSettings_MRVector2f arg) {return new(arg);}
        public _ByValue_DecimatePolylineSettings_MRVector2f(MR.Misc._Moved<DecimatePolylineSettings_MRVector2f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DecimatePolylineSettings_MRVector2f(MR.Misc._Moved<DecimatePolylineSettings_MRVector2f> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DecimatePolylineSettings_MRVector2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DecimatePolylineSettings_MRVector2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimatePolylineSettings_MRVector2f`/`Const_DecimatePolylineSettings_MRVector2f` directly.
    public class _InOptMut_DecimatePolylineSettings_MRVector2f
    {
        public DecimatePolylineSettings_MRVector2f? Opt;

        public _InOptMut_DecimatePolylineSettings_MRVector2f() {}
        public _InOptMut_DecimatePolylineSettings_MRVector2f(DecimatePolylineSettings_MRVector2f value) {Opt = value;}
        public static implicit operator _InOptMut_DecimatePolylineSettings_MRVector2f(DecimatePolylineSettings_MRVector2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `DecimatePolylineSettings_MRVector2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DecimatePolylineSettings_MRVector2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimatePolylineSettings_MRVector2f`/`Const_DecimatePolylineSettings_MRVector2f` to pass it to the function.
    public class _InOptConst_DecimatePolylineSettings_MRVector2f
    {
        public Const_DecimatePolylineSettings_MRVector2f? Opt;

        public _InOptConst_DecimatePolylineSettings_MRVector2f() {}
        public _InOptConst_DecimatePolylineSettings_MRVector2f(Const_DecimatePolylineSettings_MRVector2f value) {Opt = value;}
        public static implicit operator _InOptConst_DecimatePolylineSettings_MRVector2f(Const_DecimatePolylineSettings_MRVector2f value) {return new(value);}
    }

    /**
    * \struct MR::DecimatePolylineSettings
    * \brief Parameters structure for MR::decimatePolyline
    *
    *
    * \sa \ref decimatePolyline
    */
    /// Generated from class `MR::DecimatePolylineSettings<MR::Vector3f>`.
    /// This is the const half of the class.
    public class Const_DecimatePolylineSettings_MRVector3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DecimatePolylineSettings_MRVector3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Destroy", ExactSpelling = true)]
            extern static void __MR_DecimatePolylineSettings_MR_Vector3f_Destroy(_Underlying *_this);
            __MR_DecimatePolylineSettings_MR_Vector3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DecimatePolylineSettings_MRVector3f() {Dispose(false);}

        /// Limit from above on the maximum distance from moved vertices to original contour
        public unsafe float MaxError
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_maxError", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector3f_Get_maxError(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector3f_Get_maxError(_UnderlyingPtr);
            }
        }

        /// Maximal possible edge length created during decimation
        public unsafe float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector3f_Get_maxEdgeLen(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector3f_Get_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Stabilizer is dimensionless coefficient.
        /// The larger is stabilizer, the more Decimator will strive to retain the density of input points.
        /// If stabilizer is zero, then only the shape of input line will be preserved.
        public unsafe float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_stabilizer", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector3f_Get_stabilizer(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector3f_Get_stabilizer(_UnderlyingPtr);
            }
        }

        /// if true then after each edge collapse the position of remaining vertex is optimized to
        /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
        public unsafe bool OptimizeVertexPos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_optimizeVertexPos", ExactSpelling = true)]
                extern static bool *__MR_DecimatePolylineSettings_MR_Vector3f_Get_optimizeVertexPos(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector3f_Get_optimizeVertexPos(_UnderlyingPtr);
            }
        }

        /// Limit on the number of deleted vertices
        public unsafe int MaxDeletedVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_maxDeletedVertices", ExactSpelling = true)]
                extern static int *__MR_DecimatePolylineSettings_MR_Vector3f_Get_maxDeletedVertices(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector3f_Get_maxDeletedVertices(_UnderlyingPtr);
            }
        }

        /// Region of the polyline to be decimated, it is updated during the operation
        /// Remain nullptr to include the whole polyline
        public unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_region", ExactSpelling = true)]
                extern static void **__MR_DecimatePolylineSettings_MR_Vector3f_Get_region(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_Get_region(_UnderlyingPtr);
            }
        }

        /// Whether to allow collapsing edges with at least one vertex on the end of not-closed polyline
        /// (or on region boundary if region is given);
        /// if touchBdVertices is false then boundary vertices are strictly fixed
        public unsafe bool TouchBdVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_touchBdVertices", ExactSpelling = true)]
                extern static bool *__MR_DecimatePolylineSettings_MR_Vector3f_Get_touchBdVertices(_Underlying *_this);
                return *__MR_DecimatePolylineSettings_MR_Vector3f_Get_touchBdVertices(_UnderlyingPtr);
            }
        }

        /**
        * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
        * \details It receives the edge being collapsed: its destination vertex will disappear,
        * and its origin vertex will get new position (provided as the second argument) after collapse;
        * If the callback returns false, then the collapse is prohibited
        */
        public unsafe MR.Std.Const_Function_BoolFuncFromMREdgeIdConstMRVector3fRef PreCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_preCollapse", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_Get_preCollapse(_Underlying *_this);
                return new(__MR_DecimatePolylineSettings_MR_Vector3f_Get_preCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief The user can provide this optional callback for adjusting error introduced by this
        * edge collapse and the collapse position.
        * \details On input the callback gets the squared error and position computed by standard means,
        * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
        * This callback can be called from many threads in parallel and must be thread-safe.
        * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
        */
        public unsafe MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef AdjustCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_adjustCollapse", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_Get_adjustCollapse(_Underlying *_this);
                return new(__MR_DecimatePolylineSettings_MR_Vector3f_Get_adjustCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief  If not null, then
        * on input: if the vector is not empty then it is taken for initialization instead of form computation for all vertices;
        * on output: quadratic form for each remaining vertex is returned there
        */
        public unsafe ref void * VertForms
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_Get_vertForms", ExactSpelling = true)]
                extern static void **__MR_DecimatePolylineSettings_MR_Vector3f_Get_vertForms(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_Get_vertForms(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DecimatePolylineSettings_MRVector3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector3f._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector3f_DefaultConstruct();
        }

        /// Constructs `MR::DecimatePolylineSettings<MR::Vector3f>` elementwise.
        public unsafe Const_DecimatePolylineSettings_MRVector3f(float maxError, float maxEdgeLen, float stabilizer, bool optimizeVertexPos, int maxDeletedVertices, MR.VertBitSet? region, bool touchBdVertices, MR.Std._ByValue_Function_BoolFuncFromMREdgeIdConstMRVector3fRef preCollapse, MR.Std._ByValue_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef adjustCollapse, MR.Vector_MRQuadraticForm3f_MRVertId? vertForms) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector3f._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_ConstructFrom(float maxError, float maxEdgeLen, float stabilizer, byte optimizeVertexPos, int maxDeletedVertices, MR.VertBitSet._Underlying *region, byte touchBdVertices, MR.Misc._PassBy preCollapse_pass_by, MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *preCollapse, MR.Misc._PassBy adjustCollapse_pass_by, MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef._Underlying *adjustCollapse, MR.Vector_MRQuadraticForm3f_MRVertId._Underlying *vertForms);
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector3f_ConstructFrom(maxError, maxEdgeLen, stabilizer, optimizeVertexPos ? (byte)1 : (byte)0, maxDeletedVertices, region is not null ? region._UnderlyingPtr : null, touchBdVertices ? (byte)1 : (byte)0, preCollapse.PassByMode, preCollapse.Value is not null ? preCollapse.Value._UnderlyingPtr : null, adjustCollapse.PassByMode, adjustCollapse.Value is not null ? adjustCollapse.Value._UnderlyingPtr : null, vertForms is not null ? vertForms._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DecimatePolylineSettings<MR::Vector3f>::DecimatePolylineSettings`.
        public unsafe Const_DecimatePolylineSettings_MRVector3f(MR._ByValue_DecimatePolylineSettings_MRVector3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector3f._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DecimatePolylineSettings_MRVector3f._Underlying *_other);
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /**
    * \struct MR::DecimatePolylineSettings
    * \brief Parameters structure for MR::decimatePolyline
    *
    *
    * \sa \ref decimatePolyline
    */
    /// Generated from class `MR::DecimatePolylineSettings<MR::Vector3f>`.
    /// This is the non-const half of the class.
    public class DecimatePolylineSettings_MRVector3f : Const_DecimatePolylineSettings_MRVector3f
    {
        internal unsafe DecimatePolylineSettings_MRVector3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Limit from above on the maximum distance from moved vertices to original contour
        public new unsafe ref float MaxError
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxError", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxError(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxError(_UnderlyingPtr);
            }
        }

        /// Maximal possible edge length created during decimation
        public new unsafe ref float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxEdgeLen(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Stabilizer is dimensionless coefficient.
        /// The larger is stabilizer, the more Decimator will strive to retain the density of input points.
        /// If stabilizer is zero, then only the shape of input line will be preserved.
        public new unsafe ref float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_stabilizer", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_stabilizer(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_stabilizer(_UnderlyingPtr);
            }
        }

        /// if true then after each edge collapse the position of remaining vertex is optimized to
        /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
        public new unsafe ref bool OptimizeVertexPos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_optimizeVertexPos", ExactSpelling = true)]
                extern static bool *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_optimizeVertexPos(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_optimizeVertexPos(_UnderlyingPtr);
            }
        }

        /// Limit on the number of deleted vertices
        public new unsafe ref int MaxDeletedVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxDeletedVertices", ExactSpelling = true)]
                extern static int *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxDeletedVertices(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_maxDeletedVertices(_UnderlyingPtr);
            }
        }

        /// Region of the polyline to be decimated, it is updated during the operation
        /// Remain nullptr to include the whole polyline
        public new unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_region(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Whether to allow collapsing edges with at least one vertex on the end of not-closed polyline
        /// (or on region boundary if region is given);
        /// if touchBdVertices is false then boundary vertices are strictly fixed
        public new unsafe ref bool TouchBdVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_touchBdVertices", ExactSpelling = true)]
                extern static bool *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_touchBdVertices(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_touchBdVertices(_UnderlyingPtr);
            }
        }

        /**
        * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
        * \details It receives the edge being collapsed: its destination vertex will disappear,
        * and its origin vertex will get new position (provided as the second argument) after collapse;
        * If the callback returns false, then the collapse is prohibited
        */
        public new unsafe MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef PreCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_preCollapse", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_preCollapse(_Underlying *_this);
                return new(__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_preCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief The user can provide this optional callback for adjusting error introduced by this
        * edge collapse and the collapse position.
        * \details On input the callback gets the squared error and position computed by standard means,
        * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
        * This callback can be called from many threads in parallel and must be thread-safe.
        * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
        */
        public new unsafe MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef AdjustCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_adjustCollapse", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_adjustCollapse(_Underlying *_this);
                return new(__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_adjustCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief  If not null, then
        * on input: if the vector is not empty then it is taken for initialization instead of form computation for all vertices;
        * on output: quadratic form for each remaining vertex is returned there
        */
        public new unsafe ref void * VertForms
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_vertForms", ExactSpelling = true)]
                extern static void **__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_vertForms(_Underlying *_this);
                return ref *__MR_DecimatePolylineSettings_MR_Vector3f_GetMutable_vertForms(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DecimatePolylineSettings_MRVector3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector3f._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector3f_DefaultConstruct();
        }

        /// Constructs `MR::DecimatePolylineSettings<MR::Vector3f>` elementwise.
        public unsafe DecimatePolylineSettings_MRVector3f(float maxError, float maxEdgeLen, float stabilizer, bool optimizeVertexPos, int maxDeletedVertices, MR.VertBitSet? region, bool touchBdVertices, MR.Std._ByValue_Function_BoolFuncFromMREdgeIdConstMRVector3fRef preCollapse, MR.Std._ByValue_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef adjustCollapse, MR.Vector_MRQuadraticForm3f_MRVertId? vertForms) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector3f._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_ConstructFrom(float maxError, float maxEdgeLen, float stabilizer, byte optimizeVertexPos, int maxDeletedVertices, MR.VertBitSet._Underlying *region, byte touchBdVertices, MR.Misc._PassBy preCollapse_pass_by, MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *preCollapse, MR.Misc._PassBy adjustCollapse_pass_by, MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef._Underlying *adjustCollapse, MR.Vector_MRQuadraticForm3f_MRVertId._Underlying *vertForms);
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector3f_ConstructFrom(maxError, maxEdgeLen, stabilizer, optimizeVertexPos ? (byte)1 : (byte)0, maxDeletedVertices, region is not null ? region._UnderlyingPtr : null, touchBdVertices ? (byte)1 : (byte)0, preCollapse.PassByMode, preCollapse.Value is not null ? preCollapse.Value._UnderlyingPtr : null, adjustCollapse.PassByMode, adjustCollapse.Value is not null ? adjustCollapse.Value._UnderlyingPtr : null, vertForms is not null ? vertForms._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DecimatePolylineSettings<MR::Vector3f>::DecimatePolylineSettings`.
        public unsafe DecimatePolylineSettings_MRVector3f(MR._ByValue_DecimatePolylineSettings_MRVector3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector3f._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DecimatePolylineSettings_MRVector3f._Underlying *_other);
            _UnderlyingPtr = __MR_DecimatePolylineSettings_MR_Vector3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DecimatePolylineSettings<MR::Vector3f>::operator=`.
        public unsafe MR.DecimatePolylineSettings_MRVector3f Assign(MR._ByValue_DecimatePolylineSettings_MRVector3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineSettings_MR_Vector3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineSettings_MRVector3f._Underlying *__MR_DecimatePolylineSettings_MR_Vector3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DecimatePolylineSettings_MRVector3f._Underlying *_other);
            return new(__MR_DecimatePolylineSettings_MR_Vector3f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DecimatePolylineSettings_MRVector3f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DecimatePolylineSettings_MRVector3f`/`Const_DecimatePolylineSettings_MRVector3f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DecimatePolylineSettings_MRVector3f
    {
        internal readonly Const_DecimatePolylineSettings_MRVector3f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DecimatePolylineSettings_MRVector3f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DecimatePolylineSettings_MRVector3f(Const_DecimatePolylineSettings_MRVector3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DecimatePolylineSettings_MRVector3f(Const_DecimatePolylineSettings_MRVector3f arg) {return new(arg);}
        public _ByValue_DecimatePolylineSettings_MRVector3f(MR.Misc._Moved<DecimatePolylineSettings_MRVector3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DecimatePolylineSettings_MRVector3f(MR.Misc._Moved<DecimatePolylineSettings_MRVector3f> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DecimatePolylineSettings_MRVector3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DecimatePolylineSettings_MRVector3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimatePolylineSettings_MRVector3f`/`Const_DecimatePolylineSettings_MRVector3f` directly.
    public class _InOptMut_DecimatePolylineSettings_MRVector3f
    {
        public DecimatePolylineSettings_MRVector3f? Opt;

        public _InOptMut_DecimatePolylineSettings_MRVector3f() {}
        public _InOptMut_DecimatePolylineSettings_MRVector3f(DecimatePolylineSettings_MRVector3f value) {Opt = value;}
        public static implicit operator _InOptMut_DecimatePolylineSettings_MRVector3f(DecimatePolylineSettings_MRVector3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `DecimatePolylineSettings_MRVector3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DecimatePolylineSettings_MRVector3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimatePolylineSettings_MRVector3f`/`Const_DecimatePolylineSettings_MRVector3f` to pass it to the function.
    public class _InOptConst_DecimatePolylineSettings_MRVector3f
    {
        public Const_DecimatePolylineSettings_MRVector3f? Opt;

        public _InOptConst_DecimatePolylineSettings_MRVector3f() {}
        public _InOptConst_DecimatePolylineSettings_MRVector3f(Const_DecimatePolylineSettings_MRVector3f value) {Opt = value;}
        public static implicit operator _InOptConst_DecimatePolylineSettings_MRVector3f(Const_DecimatePolylineSettings_MRVector3f value) {return new(value);}
    }

    /**
    * \struct MR::DecimatePolylineResult
    * \brief Results of MR::decimateContour
    */
    /// Generated from class `MR::DecimatePolylineResult`.
    /// This is the const half of the class.
    public class Const_DecimatePolylineResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DecimatePolylineResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_Destroy", ExactSpelling = true)]
            extern static void __MR_DecimatePolylineResult_Destroy(_Underlying *_this);
            __MR_DecimatePolylineResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DecimatePolylineResult() {Dispose(false);}

        ///< Number deleted verts. Same as the number of performed collapses
        public unsafe int VertsDeleted
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_Get_vertsDeleted", ExactSpelling = true)]
                extern static int *__MR_DecimatePolylineResult_Get_vertsDeleted(_Underlying *_this);
                return *__MR_DecimatePolylineResult_Get_vertsDeleted(_UnderlyingPtr);
            }
        }

        ///< Max different (as distance) between original contour and result contour
        public unsafe float ErrorIntroduced
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_Get_errorIntroduced", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineResult_Get_errorIntroduced(_Underlying *_this);
                return *__MR_DecimatePolylineResult_Get_errorIntroduced(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DecimatePolylineResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimatePolylineResult._Underlying *__MR_DecimatePolylineResult_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimatePolylineResult_DefaultConstruct();
        }

        /// Constructs `MR::DecimatePolylineResult` elementwise.
        public unsafe Const_DecimatePolylineResult(int vertsDeleted, float errorIntroduced) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimatePolylineResult._Underlying *__MR_DecimatePolylineResult_ConstructFrom(int vertsDeleted, float errorIntroduced);
            _UnderlyingPtr = __MR_DecimatePolylineResult_ConstructFrom(vertsDeleted, errorIntroduced);
        }

        /// Generated from constructor `MR::DecimatePolylineResult::DecimatePolylineResult`.
        public unsafe Const_DecimatePolylineResult(MR.Const_DecimatePolylineResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineResult._Underlying *__MR_DecimatePolylineResult_ConstructFromAnother(MR.DecimatePolylineResult._Underlying *_other);
            _UnderlyingPtr = __MR_DecimatePolylineResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /**
    * \struct MR::DecimatePolylineResult
    * \brief Results of MR::decimateContour
    */
    /// Generated from class `MR::DecimatePolylineResult`.
    /// This is the non-const half of the class.
    public class DecimatePolylineResult : Const_DecimatePolylineResult
    {
        internal unsafe DecimatePolylineResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< Number deleted verts. Same as the number of performed collapses
        public new unsafe ref int VertsDeleted
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_GetMutable_vertsDeleted", ExactSpelling = true)]
                extern static int *__MR_DecimatePolylineResult_GetMutable_vertsDeleted(_Underlying *_this);
                return ref *__MR_DecimatePolylineResult_GetMutable_vertsDeleted(_UnderlyingPtr);
            }
        }

        ///< Max different (as distance) between original contour and result contour
        public new unsafe ref float ErrorIntroduced
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_GetMutable_errorIntroduced", ExactSpelling = true)]
                extern static float *__MR_DecimatePolylineResult_GetMutable_errorIntroduced(_Underlying *_this);
                return ref *__MR_DecimatePolylineResult_GetMutable_errorIntroduced(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DecimatePolylineResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimatePolylineResult._Underlying *__MR_DecimatePolylineResult_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimatePolylineResult_DefaultConstruct();
        }

        /// Constructs `MR::DecimatePolylineResult` elementwise.
        public unsafe DecimatePolylineResult(int vertsDeleted, float errorIntroduced) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimatePolylineResult._Underlying *__MR_DecimatePolylineResult_ConstructFrom(int vertsDeleted, float errorIntroduced);
            _UnderlyingPtr = __MR_DecimatePolylineResult_ConstructFrom(vertsDeleted, errorIntroduced);
        }

        /// Generated from constructor `MR::DecimatePolylineResult::DecimatePolylineResult`.
        public unsafe DecimatePolylineResult(MR.Const_DecimatePolylineResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineResult._Underlying *__MR_DecimatePolylineResult_ConstructFromAnother(MR.DecimatePolylineResult._Underlying *_other);
            _UnderlyingPtr = __MR_DecimatePolylineResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::DecimatePolylineResult::operator=`.
        public unsafe MR.DecimatePolylineResult Assign(MR.Const_DecimatePolylineResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimatePolylineResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DecimatePolylineResult._Underlying *__MR_DecimatePolylineResult_AssignFromAnother(_Underlying *_this, MR.DecimatePolylineResult._Underlying *_other);
            return new(__MR_DecimatePolylineResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `DecimatePolylineResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DecimatePolylineResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimatePolylineResult`/`Const_DecimatePolylineResult` directly.
    public class _InOptMut_DecimatePolylineResult
    {
        public DecimatePolylineResult? Opt;

        public _InOptMut_DecimatePolylineResult() {}
        public _InOptMut_DecimatePolylineResult(DecimatePolylineResult value) {Opt = value;}
        public static implicit operator _InOptMut_DecimatePolylineResult(DecimatePolylineResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `DecimatePolylineResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DecimatePolylineResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimatePolylineResult`/`Const_DecimatePolylineResult` to pass it to the function.
    public class _InOptConst_DecimatePolylineResult
    {
        public Const_DecimatePolylineResult? Opt;

        public _InOptConst_DecimatePolylineResult() {}
        public _InOptConst_DecimatePolylineResult(Const_DecimatePolylineResult value) {Opt = value;}
        public static implicit operator _InOptConst_DecimatePolylineResult(Const_DecimatePolylineResult value) {return new(value);}
    }

    /**
    * \brief Collapse edges in the polyline according to the settings
    *
    */
    /// Generated from function `MR::decimatePolyline`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.DecimatePolylineResult DecimatePolyline(MR.Polyline2 polyline, MR.Const_DecimatePolylineSettings_MRVector2f? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decimatePolyline_MR_Polyline2", ExactSpelling = true)]
        extern static MR.DecimatePolylineResult._Underlying *__MR_decimatePolyline_MR_Polyline2(MR.Polyline2._Underlying *polyline, MR.Const_DecimatePolylineSettings_MRVector2f._Underlying *settings);
        return new(__MR_decimatePolyline_MR_Polyline2(polyline._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true);
    }

    /// Generated from function `MR::decimatePolyline`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.DecimatePolylineResult DecimatePolyline(MR.Polyline3 polyline, MR.Const_DecimatePolylineSettings_MRVector3f? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decimatePolyline_MR_Polyline3", ExactSpelling = true)]
        extern static MR.DecimatePolylineResult._Underlying *__MR_decimatePolyline_MR_Polyline3(MR.Polyline3._Underlying *polyline, MR.Const_DecimatePolylineSettings_MRVector3f._Underlying *settings);
        return new(__MR_decimatePolyline_MR_Polyline3(polyline._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true);
    }

    /**
    * \brief Collapse edges in the contour according to the settings
    *
    */
    /// Generated from function `MR::decimateContour`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.DecimatePolylineResult DecimateContour(MR.Std.Vector_MRVector2f contour, MR.Const_DecimatePolylineSettings_MRVector2f? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decimateContour_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.DecimatePolylineResult._Underlying *__MR_decimateContour_std_vector_MR_Vector2f(MR.Std.Vector_MRVector2f._Underlying *contour, MR.Const_DecimatePolylineSettings_MRVector2f._Underlying *settings);
        return new(__MR_decimateContour_std_vector_MR_Vector2f(contour._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true);
    }

    /// Generated from function `MR::decimateContour`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.DecimatePolylineResult DecimateContour(MR.Std.Vector_MRVector3f contour, MR.Const_DecimatePolylineSettings_MRVector3f? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decimateContour_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.DecimatePolylineResult._Underlying *__MR_decimateContour_std_vector_MR_Vector3f(MR.Std.Vector_MRVector3f._Underlying *contour, MR.Const_DecimatePolylineSettings_MRVector3f._Underlying *settings);
        return new(__MR_decimateContour_std_vector_MR_Vector3f(contour._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true);
    }
}

public static partial class MR
{
    /// the class models water increase in the terrain under the rain with constant precipitation
    /// Generated from class `MR::PrecipitationSimulator`.
    /// This is the const half of the class.
    public class Const_PrecipitationSimulator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PrecipitationSimulator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_Destroy", ExactSpelling = true)]
            extern static void __MR_PrecipitationSimulator_Destroy(_Underlying *_this);
            __MR_PrecipitationSimulator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PrecipitationSimulator() {Dispose(false);}

        /// Generated from constructor `MR::PrecipitationSimulator::PrecipitationSimulator`.
        public unsafe Const_PrecipitationSimulator(MR._ByValue_PrecipitationSimulator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PrecipitationSimulator._Underlying *__MR_PrecipitationSimulator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PrecipitationSimulator._Underlying *_other);
            _UnderlyingPtr = __MR_PrecipitationSimulator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// initializes modeling from the initial subdivision of the terrain
        /// Generated from constructor `MR::PrecipitationSimulator::PrecipitationSimulator`.
        public unsafe Const_PrecipitationSimulator(MR.WatershedGraph wg) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_Construct", ExactSpelling = true)]
            extern static MR.PrecipitationSimulator._Underlying *__MR_PrecipitationSimulator_Construct(MR.WatershedGraph._Underlying *wg);
            _UnderlyingPtr = __MR_PrecipitationSimulator_Construct(wg._UnderlyingPtr);
        }

        /// initializes modeling from the initial subdivision of the terrain
        /// Generated from constructor `MR::PrecipitationSimulator::PrecipitationSimulator`.
        public static unsafe implicit operator Const_PrecipitationSimulator(MR.WatershedGraph wg) {return new(wg);}

        public enum Event : int
        {
            ///< all basins are full and water goes outside
            Finish = 0,
            ///< one basin just became full
            BasinFull = 1,
            ///< two basins just merged
            Merge = 2,
        }

        /// Generated from class `MR::PrecipitationSimulator::SimulationStep`.
        /// This is the const half of the class.
        public class Const_SimulationStep : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_SimulationStep(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_Destroy", ExactSpelling = true)]
                extern static void __MR_PrecipitationSimulator_SimulationStep_Destroy(_Underlying *_this);
                __MR_PrecipitationSimulator_SimulationStep_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_SimulationStep() {Dispose(false);}

            public unsafe MR.PrecipitationSimulator.Event Event
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_Get_event", ExactSpelling = true)]
                    extern static MR.PrecipitationSimulator.Event *__MR_PrecipitationSimulator_SimulationStep_Get_event(_Underlying *_this);
                    return *__MR_PrecipitationSimulator_SimulationStep_Get_event(_UnderlyingPtr);
                }
            }

            ///< amount of precipitation (in same units as mesh coordinates and water level)
            public unsafe float Amount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_Get_amount", ExactSpelling = true)]
                    extern static float *__MR_PrecipitationSimulator_SimulationStep_Get_amount(_Underlying *_this);
                    return *__MR_PrecipitationSimulator_SimulationStep_Get_amount(_UnderlyingPtr);
                }
            }

            ///< BasinFull: this basin just became full
            ///< Merge: this basin just absorbed the other basin
            public unsafe MR.Const_GraphVertId Basin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_Get_basin", ExactSpelling = true)]
                    extern static MR.Const_GraphVertId._Underlying *__MR_PrecipitationSimulator_SimulationStep_Get_basin(_Underlying *_this);
                    return new(__MR_PrecipitationSimulator_SimulationStep_Get_basin(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< BasinFull: the flow from full basin will first go here (may be not the last destination)
            ///< Merge: this basin was just absorbed
            public unsafe MR.Const_GraphVertId NeiBasin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_Get_neiBasin", ExactSpelling = true)]
                    extern static MR.Const_GraphVertId._Underlying *__MR_PrecipitationSimulator_SimulationStep_Get_neiBasin(_Underlying *_this);
                    return new(__MR_PrecipitationSimulator_SimulationStep_Get_neiBasin(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_SimulationStep() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PrecipitationSimulator.SimulationStep._Underlying *__MR_PrecipitationSimulator_SimulationStep_DefaultConstruct();
                _UnderlyingPtr = __MR_PrecipitationSimulator_SimulationStep_DefaultConstruct();
            }

            /// Constructs `MR::PrecipitationSimulator::SimulationStep` elementwise.
            public unsafe Const_SimulationStep(MR.PrecipitationSimulator.Event event_, float amount, MR.GraphVertId basin, MR.GraphVertId neiBasin) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_ConstructFrom", ExactSpelling = true)]
                extern static MR.PrecipitationSimulator.SimulationStep._Underlying *__MR_PrecipitationSimulator_SimulationStep_ConstructFrom(MR.PrecipitationSimulator.Event event_, float amount, MR.GraphVertId basin, MR.GraphVertId neiBasin);
                _UnderlyingPtr = __MR_PrecipitationSimulator_SimulationStep_ConstructFrom(event_, amount, basin, neiBasin);
            }

            /// Generated from constructor `MR::PrecipitationSimulator::SimulationStep::SimulationStep`.
            public unsafe Const_SimulationStep(MR.PrecipitationSimulator.Const_SimulationStep _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PrecipitationSimulator.SimulationStep._Underlying *__MR_PrecipitationSimulator_SimulationStep_ConstructFromAnother(MR.PrecipitationSimulator.SimulationStep._Underlying *_other);
                _UnderlyingPtr = __MR_PrecipitationSimulator_SimulationStep_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::PrecipitationSimulator::SimulationStep`.
        /// This is the non-const half of the class.
        public class SimulationStep : Const_SimulationStep
        {
            internal unsafe SimulationStep(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe ref MR.PrecipitationSimulator.Event Event
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_GetMutable_event", ExactSpelling = true)]
                    extern static MR.PrecipitationSimulator.Event *__MR_PrecipitationSimulator_SimulationStep_GetMutable_event(_Underlying *_this);
                    return ref *__MR_PrecipitationSimulator_SimulationStep_GetMutable_event(_UnderlyingPtr);
                }
            }

            ///< amount of precipitation (in same units as mesh coordinates and water level)
            public new unsafe ref float Amount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_GetMutable_amount", ExactSpelling = true)]
                    extern static float *__MR_PrecipitationSimulator_SimulationStep_GetMutable_amount(_Underlying *_this);
                    return ref *__MR_PrecipitationSimulator_SimulationStep_GetMutable_amount(_UnderlyingPtr);
                }
            }

            ///< BasinFull: this basin just became full
            ///< Merge: this basin just absorbed the other basin
            public new unsafe MR.Mut_GraphVertId Basin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_GetMutable_basin", ExactSpelling = true)]
                    extern static MR.Mut_GraphVertId._Underlying *__MR_PrecipitationSimulator_SimulationStep_GetMutable_basin(_Underlying *_this);
                    return new(__MR_PrecipitationSimulator_SimulationStep_GetMutable_basin(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< BasinFull: the flow from full basin will first go here (may be not the last destination)
            ///< Merge: this basin was just absorbed
            public new unsafe MR.Mut_GraphVertId NeiBasin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_GetMutable_neiBasin", ExactSpelling = true)]
                    extern static MR.Mut_GraphVertId._Underlying *__MR_PrecipitationSimulator_SimulationStep_GetMutable_neiBasin(_Underlying *_this);
                    return new(__MR_PrecipitationSimulator_SimulationStep_GetMutable_neiBasin(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe SimulationStep() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PrecipitationSimulator.SimulationStep._Underlying *__MR_PrecipitationSimulator_SimulationStep_DefaultConstruct();
                _UnderlyingPtr = __MR_PrecipitationSimulator_SimulationStep_DefaultConstruct();
            }

            /// Constructs `MR::PrecipitationSimulator::SimulationStep` elementwise.
            public unsafe SimulationStep(MR.PrecipitationSimulator.Event event_, float amount, MR.GraphVertId basin, MR.GraphVertId neiBasin) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_ConstructFrom", ExactSpelling = true)]
                extern static MR.PrecipitationSimulator.SimulationStep._Underlying *__MR_PrecipitationSimulator_SimulationStep_ConstructFrom(MR.PrecipitationSimulator.Event event_, float amount, MR.GraphVertId basin, MR.GraphVertId neiBasin);
                _UnderlyingPtr = __MR_PrecipitationSimulator_SimulationStep_ConstructFrom(event_, amount, basin, neiBasin);
            }

            /// Generated from constructor `MR::PrecipitationSimulator::SimulationStep::SimulationStep`.
            public unsafe SimulationStep(MR.PrecipitationSimulator.Const_SimulationStep _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PrecipitationSimulator.SimulationStep._Underlying *__MR_PrecipitationSimulator_SimulationStep_ConstructFromAnother(MR.PrecipitationSimulator.SimulationStep._Underlying *_other);
                _UnderlyingPtr = __MR_PrecipitationSimulator_SimulationStep_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::PrecipitationSimulator::SimulationStep::operator=`.
            public unsafe MR.PrecipitationSimulator.SimulationStep Assign(MR.PrecipitationSimulator.Const_SimulationStep _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_SimulationStep_AssignFromAnother", ExactSpelling = true)]
                extern static MR.PrecipitationSimulator.SimulationStep._Underlying *__MR_PrecipitationSimulator_SimulationStep_AssignFromAnother(_Underlying *_this, MR.PrecipitationSimulator.SimulationStep._Underlying *_other);
                return new(__MR_PrecipitationSimulator_SimulationStep_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `SimulationStep` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_SimulationStep`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SimulationStep`/`Const_SimulationStep` directly.
        public class _InOptMut_SimulationStep
        {
            public SimulationStep? Opt;

            public _InOptMut_SimulationStep() {}
            public _InOptMut_SimulationStep(SimulationStep value) {Opt = value;}
            public static implicit operator _InOptMut_SimulationStep(SimulationStep value) {return new(value);}
        }

        /// This is used for optional parameters of class `SimulationStep` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_SimulationStep`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SimulationStep`/`Const_SimulationStep` to pass it to the function.
        public class _InOptConst_SimulationStep
        {
            public Const_SimulationStep? Opt;

            public _InOptConst_SimulationStep() {}
            public _InOptConst_SimulationStep(Const_SimulationStep value) {Opt = value;}
            public static implicit operator _InOptConst_SimulationStep(Const_SimulationStep value) {return new(value);}
        }
    }

    /// the class models water increase in the terrain under the rain with constant precipitation
    /// Generated from class `MR::PrecipitationSimulator`.
    /// This is the non-const half of the class.
    public class PrecipitationSimulator : Const_PrecipitationSimulator
    {
        internal unsafe PrecipitationSimulator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::PrecipitationSimulator::PrecipitationSimulator`.
        public unsafe PrecipitationSimulator(MR._ByValue_PrecipitationSimulator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PrecipitationSimulator._Underlying *__MR_PrecipitationSimulator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PrecipitationSimulator._Underlying *_other);
            _UnderlyingPtr = __MR_PrecipitationSimulator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// initializes modeling from the initial subdivision of the terrain
        /// Generated from constructor `MR::PrecipitationSimulator::PrecipitationSimulator`.
        public unsafe PrecipitationSimulator(MR.WatershedGraph wg) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_Construct", ExactSpelling = true)]
            extern static MR.PrecipitationSimulator._Underlying *__MR_PrecipitationSimulator_Construct(MR.WatershedGraph._Underlying *wg);
            _UnderlyingPtr = __MR_PrecipitationSimulator_Construct(wg._UnderlyingPtr);
        }

        /// initializes modeling from the initial subdivision of the terrain
        /// Generated from constructor `MR::PrecipitationSimulator::PrecipitationSimulator`.
        public static unsafe implicit operator PrecipitationSimulator(MR.WatershedGraph wg) {return new(wg);}

        /// processes the next event happened with the terrain basins
        /// Generated from method `MR::PrecipitationSimulator::simulateOne`.
        public unsafe MR.PrecipitationSimulator.SimulationStep SimulateOne()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PrecipitationSimulator_simulateOne", ExactSpelling = true)]
            extern static MR.PrecipitationSimulator.SimulationStep._Underlying *__MR_PrecipitationSimulator_simulateOne(_Underlying *_this);
            return new(__MR_PrecipitationSimulator_simulateOne(_UnderlyingPtr), is_owning: true);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PrecipitationSimulator` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PrecipitationSimulator`/`Const_PrecipitationSimulator` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PrecipitationSimulator
    {
        internal readonly Const_PrecipitationSimulator? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PrecipitationSimulator(Const_PrecipitationSimulator new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PrecipitationSimulator(Const_PrecipitationSimulator arg) {return new(arg);}
        public _ByValue_PrecipitationSimulator(MR.Misc._Moved<PrecipitationSimulator> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PrecipitationSimulator(MR.Misc._Moved<PrecipitationSimulator> arg) {return new(arg);}

        /// initializes modeling from the initial subdivision of the terrain
        /// Generated from constructor `MR::PrecipitationSimulator::PrecipitationSimulator`.
        public static unsafe implicit operator _ByValue_PrecipitationSimulator(MR.WatershedGraph wg) {return new MR.PrecipitationSimulator(wg);}
    }

    /// This is used for optional parameters of class `PrecipitationSimulator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PrecipitationSimulator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PrecipitationSimulator`/`Const_PrecipitationSimulator` directly.
    public class _InOptMut_PrecipitationSimulator
    {
        public PrecipitationSimulator? Opt;

        public _InOptMut_PrecipitationSimulator() {}
        public _InOptMut_PrecipitationSimulator(PrecipitationSimulator value) {Opt = value;}
        public static implicit operator _InOptMut_PrecipitationSimulator(PrecipitationSimulator value) {return new(value);}
    }

    /// This is used for optional parameters of class `PrecipitationSimulator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PrecipitationSimulator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PrecipitationSimulator`/`Const_PrecipitationSimulator` to pass it to the function.
    public class _InOptConst_PrecipitationSimulator
    {
        public Const_PrecipitationSimulator? Opt;

        public _InOptConst_PrecipitationSimulator() {}
        public _InOptConst_PrecipitationSimulator(Const_PrecipitationSimulator value) {Opt = value;}
        public static implicit operator _InOptConst_PrecipitationSimulator(Const_PrecipitationSimulator value) {return new(value);}

        /// initializes modeling from the initial subdivision of the terrain
        /// Generated from constructor `MR::PrecipitationSimulator::PrecipitationSimulator`.
        public static unsafe implicit operator _InOptConst_PrecipitationSimulator(MR.WatershedGraph wg) {return new MR.PrecipitationSimulator(wg);}
    }
}

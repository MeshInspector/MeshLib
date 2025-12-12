public static partial class MR
{
    /// safely invokes \param cb with given value; just returning true for empty callback
    /// Generated from function `MR::reportProgress`.
    public static unsafe bool ReportProgress(MR.Std._ByValue_Function_BoolFuncFromFloat cb, float v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_reportProgress_2", ExactSpelling = true)]
        extern static byte __MR_reportProgress_2(MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, float v);
        return __MR_reportProgress_2(cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, v) != 0;
    }

    /// safely invokes \param cb with given value if \param counter is divisible by \param divider (preferably a power of 2);
    /// just returning true for empty callback
    /// Generated from function `MR::reportProgress`.
    public static unsafe bool ReportProgress(MR.Std._ByValue_Function_BoolFuncFromFloat cb, float v, ulong counter, int divider)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_reportProgress_4", ExactSpelling = true)]
        extern static byte __MR_reportProgress_4(MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, float v, ulong counter, int divider);
        return __MR_reportProgress_4(cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, v, counter, divider) != 0;
    }

    /// returns a callback that maps [0,1] linearly into [from,to] in the call to \param cb (which can be empty)
    /// Generated from function `MR::subprogress`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_BoolFuncFromFloat> Subprogress(MR.Std._ByValue_Function_BoolFuncFromFloat cb, float from, float to)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subprogress_float", ExactSpelling = true)]
        extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_subprogress_float(MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, float from, float to);
        return MR.Misc.Move(new MR.Std.Function_BoolFuncFromFloat(__MR_subprogress_float(cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, from, to), is_owning: true));
    }

    /// returns a callback that maps [0,1] linearly into [(index+0)/count,(index+1)/count] in the call to \param cb (which can be empty)
    /// Generated from function `MR::subprogress`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_BoolFuncFromFloat> Subprogress(MR.Std._ByValue_Function_BoolFuncFromFloat cb, ulong index, ulong count)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subprogress_uint64_t", ExactSpelling = true)]
        extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_subprogress_uint64_t(MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, ulong index, ulong count);
        return MR.Misc.Move(new MR.Std.Function_BoolFuncFromFloat(__MR_subprogress_uint64_t(cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, index, count), is_owning: true));
    }
}

public static partial class MR
{
    // Makes square plane 1x1 size with center at (0,0,0) and (0,0,1) normal
    /// Generated from function `MR::makePlane`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakePlane()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makePlane", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makePlane();
        return MR.Misc.Move(new MR.Mesh(__MR_makePlane(), is_owning: true));
    }
}

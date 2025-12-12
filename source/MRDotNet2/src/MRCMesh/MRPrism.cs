public static partial class MR
{
    //Creates a triangular prism. One edge of its base lies on 'x' axis and has 'baseLength' in length. 
    //'leftAngle' and 'rightAngle' specify two adjacent angles
    // axis of a prism is parallel to 'z' axis
    /// Generated from function `MR::makePrism`.
    /// Parameter `height` defaults to `1.0f`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakePrism(float baseLength, float leftAngle, float rightAngle, float? height = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makePrism", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makePrism(float baseLength, float leftAngle, float rightAngle, float *height);
        float __deref_height = height.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makePrism(baseLength, leftAngle, rightAngle, height.HasValue ? &__deref_height : null), is_owning: true));
    }
}

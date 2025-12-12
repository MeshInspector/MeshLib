public static partial class MR
{
    public static partial class ImageTransform
    {
        /// creates an image rotated 90 degrees clockwise
        /// Generated from function `MR::ImageTransform::rotateClockwise90`.
        public static unsafe MR.Misc._Moved<MR.Image> RotateClockwise90(MR.Const_Image image)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageTransform_rotateClockwise90", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_ImageTransform_rotateClockwise90(MR.Const_Image._Underlying *image);
            return MR.Misc.Move(new MR.Image(__MR_ImageTransform_rotateClockwise90(image._UnderlyingPtr), is_owning: true));
        }
    }
}

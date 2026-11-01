import meshlib.Functions;
import meshlib.Mesh;
import meshlib.NoiseSettings;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class NoiseDenoise
{
    public static void main(String[] args)
    {
        // Load mesh. An error is thrown as `UnexpectedResultException`.
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);

        // Add noise to the mesh
        NoiseSettings settings = new NoiseSettings();
        settings.setSigma(mesh.computeBoundingBox(null).diagonal() * 0.0001f);
        Functions.addNoise(mesh.points(), mesh.topology().getValidVerts(), settings);

        // Invalidate the mesh because of the external vertex changes
        mesh.invalidateCaches(null);

        // Save the noised mesh
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "noised_mesh.stl", null);

        // Denoise the mesh with sharpening for sharp edges
        // see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
        Functions.meshDenoiseViaNormals(mesh, null);

        // Save the denoised mesh
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "denoised_mesh.stl", null);
    }
}

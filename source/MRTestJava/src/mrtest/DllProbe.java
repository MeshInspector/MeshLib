package mrtest;

import java.io.File;
import java.util.Arrays;

/** Tries to load every shared library in a directory, to find the one whose dependencies are broken. Usage: `java mrtest.DllProbe <dir>`. */
public final class DllProbe
{
    public static void main(String[] args)
    {
        File[] files = new File(args[0]).listFiles((dir, name) -> name.endsWith(".dll") || name.endsWith(".so") || name.endsWith(".dylib"));
        Arrays.sort(files);
        int failed = 0;
        for (File file : files)
        {
            try
            {
                System.load(file.getAbsolutePath());
            }
            catch (Throwable e)
            {
                failed++;
                System.out.println("FAILED " + file.getName() + ": " + e.getMessage());
            }
        }
        System.out.println(files.length + " libraries, " + failed + " failed to load");
    }
}

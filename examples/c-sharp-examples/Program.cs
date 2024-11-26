using System.Reflection;

internal class Program
{
    static void Main(string[] args)
    {
        var assembly = Assembly.GetExecutingAssembly();
        var types = assembly.GetTypes();
        var exampleNames = types.Select(t => t.Name).Where(t => t.EndsWith("Example"));

        if (args.Length < 1 || exampleNames.Contains(args[0]) == false )
        {
            Console.WriteLine("Usage: {0} EXAMPLE_NAME [ARGS...]", Assembly.GetExecutingAssembly().GetName().Name);
            Console.WriteLine("Available examples:");
            foreach (var exampleName in exampleNames)
            {
                Console.WriteLine("\t{0}", exampleName);
            }
            return;
        }        

        foreach (var type in types)
        {
            if (type.Name == args[0])
            {
                MethodInfo runMethod = type.GetMethod("Run", BindingFlags.Static | BindingFlags.Public);
                if (runMethod == null)
                {
                    Console.WriteLine($"Run Method not found in {type.Name}.");
                    return;
                }

                runMethod.Invoke(null, new object[] { args });
            }
        }
    }
}

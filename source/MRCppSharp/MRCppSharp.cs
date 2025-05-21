using CppSharp;
using CppSharp.AST;
using CppSharp.Generators;

using Microsoft.Extensions.FileSystemGlobbing;
using Microsoft.Extensions.FileSystemGlobbing.Abstractions;

ConsoleDriver.Run(new MRCppSharp());

public class MRCppSharp : ILibrary
{
    /// Setup the driver options here.
    public void Setup(Driver driver)
    {
        var options = driver.Options;
        options.GeneratorKind = GeneratorKind.CSharp;
        options.CheckSymbols = true;
        options.GenerateClassTemplates = true;

        driver.ParserOptions.LanguageVersion = CppSharp.Parser.LanguageVersion.CPP23;
        driver.ParserOptions.AddIncludeDirs("../../../../../thirdparty/parallel-hashmap");

        var module = options.AddModule("MRMeshCppSharp");

        // Find the headers.
        string include_dir = "../../../../MRMesh";
        module.IncludeDirs.Add(include_dir);
        Matcher include_matcher = new();
        include_matcher.AddInclude("*.h");
        PatternMatchingResult include_matches = include_matcher.Execute(new DirectoryInfoWrapper(new DirectoryInfo(include_dir)));

        var i = 0;
        foreach (var match in include_matches.Files)
        {
            i++;
            if (i > 10)
                break; // <---- Delete me.  Here we artifically limit the number of parsed files.

            var filename = Path.GetFileName(match.Path);
            Console.WriteLine("Globbed header: {0}", filename);
            module.Headers.Add(filename);
        }

        // Include search paths.
        module.IncludeDirs.Add("../../../..");

        module.LibraryDirs.Add("../../../../x64/Debug");

        module.Libraries.Add("MRMesh");
    }

    /// Setup your passes here.
    public void SetupPasses(Driver driver) { }

    /// Do transformations that should happen before passes are processed.
    public void Preprocess(Driver driver, ASTContext ctx) { }

    /// Do transformations that should happen after passes are processed.
    public void Postprocess(Driver driver, ASTContext ctx) { }
}

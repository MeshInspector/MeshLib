$submodules = @( 'thirdparty/imgui', 'thirdparty/mrbind' )

git -C "$PSScriptRoot\.." submodule update --init --force --depth 1 @submodules
if ( $LASTEXITCODE -ne 0 ) { exit $LASTEXITCODE }

git -C "$PSScriptRoot\..\thirdparty\mrbind" submodule update --init --force --depth 1 deps/cppdecl
if ( $LASTEXITCODE -ne 0 ) { exit $LASTEXITCODE }

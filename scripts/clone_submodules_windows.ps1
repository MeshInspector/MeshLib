# --wheel: also mrbind-pybind11, whose sources the wheel shims are compiled from
$submodules = @( 'thirdparty/imgui', 'thirdparty/mrbind' )
if ( $args -contains '--wheel' ) { $submodules += 'thirdparty/mrbind-pybind11' }

git -C "$PSScriptRoot\.." submodule update --init --force --depth 1 @submodules
if ( $LASTEXITCODE -ne 0 ) { exit $LASTEXITCODE }

git -C "$PSScriptRoot\..\thirdparty\mrbind" submodule update --init --force --depth 1 deps/cppdecl
if ( $LASTEXITCODE -ne 0 ) { exit $LASTEXITCODE }

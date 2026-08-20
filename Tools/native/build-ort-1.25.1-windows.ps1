param(
    [Parameter(Mandatory = $true)][ValidateSet("DirectML", "CUDA")][string]$Provider,
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    [string]$CudnnHome = $CudaHome
)

$ErrorActionPreference = "Stop"
$expectedCommit = "8a77e459420f58fb946fd9067285cfa719f10bdd"
$actualCommit = (& git -C $Source rev-parse HEAD).Trim()
if ($actualCommit -ne $expectedCommit) {
    throw "Expected ONNX Runtime v1.25.1 commit $expectedCommit; found $actualCommit."
}

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$common = @("--config", "Release", "--build_shared_lib", "--parallel", "--skip_tests", "--update", "--build")
if ($Provider -eq "DirectML") {
    & (Join-Path $Source "build.bat") @common --use_dml
    $build = Join-Path $Source "build\Windows\Release"
    $ort = Get-ChildItem -LiteralPath $build -Recurse -Filter "onnxruntime.dll" |
        Where-Object { $_.FullName -notmatch "test|nuget" } | Select-Object -First 1
    $directMl = Get-ChildItem -LiteralPath $build -Recurse -Filter "DirectML.dll" | Select-Object -First 1
    if ($null -eq $ort -or $null -eq $directMl) { throw "DirectML build outputs were not found below $build." }
    Copy-Item -LiteralPath $ort.FullName -Destination $Output -Force
    Copy-Item -LiteralPath $directMl.FullName -Destination $Output -Force
} else {
    # TODO(provider package split): move this source build and its toolchain image to
    # an independently versioned CUDA provider build once the plug-in ABI is stable.
    & (Join-Path $Source "build.bat") @common --use_cuda --cuda_home $CudaHome --cudnn_home $CudnnHome `
        --cmake_extra_defines onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON
    $cuda = Get-ChildItem -LiteralPath (Join-Path $Source "build\Windows\Release") -Recurse `
        -Filter "onnxruntime_providers_cuda.dll" | Select-Object -First 1
    if ($null -eq $cuda) { throw "CUDA provider build output was not found." }
    Copy-Item -LiteralPath $cuda.FullName -Destination $Output -Force
}

Get-ChildItem -LiteralPath $Output -File | ForEach-Object {
    "{0}  {1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.Name
}

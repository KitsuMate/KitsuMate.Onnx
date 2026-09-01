#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.Build.Profile;
using UnityEditorInternal;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    internal enum OnnxProviderReadinessKind
    {
        Checking,
        Ready,
        NotConfigured,
        Unsupported,
        NotIncluded,
        UnavailableInRuntime,
        RegistrationFailed,
        MissingPayload,
        MissingNativeDependencies,
        UnsupportedHardware,
        InvalidDevice
    }

    internal readonly struct OnnxProviderReadiness
    {
        internal OnnxProviderReadiness(OnnxProviderReadinessKind kind, string text, string reason)
        {
            Kind = kind;
            Text = text;
            Reason = reason;
        }

        internal OnnxProviderReadinessKind Kind { get; }
        internal string Text { get; }
        internal string Reason { get; }
    }

    internal static class OnnxProviderReadinessEvaluator
    {
        internal static OnnxProviderReadiness EvaluateEditor(
            OnnxExecutionProvider provider,
            bool configured,
            RuntimePlatform platform,
            bool included,
            bool runtimeAvailable,
            bool probeComplete,
            string probeFailure,
            string registrationFailure,
            string runtimeFailure)
        {
            if (!configured)
                return State(OnnxProviderReadinessKind.NotConfigured, "Not configured", "This provider is not present in Provider Order.");
            if (!OnnxRuntimeBackend.IsProviderValidForPlatform(provider, platform))
                return State(OnnxProviderReadinessKind.Unsupported, "Unsupported", $"{DisplayName(provider)} cannot run on {platform}.");
            if (!included)
                return State(OnnxProviderReadinessKind.NotIncluded, "Not included", "The provider is disabled by the current Editor compilation defines.");
            if (!string.IsNullOrWhiteSpace(runtimeFailure))
                return State(OnnxProviderReadinessKind.UnavailableInRuntime, "Runtime unavailable", runtimeFailure);
            if (!string.IsNullOrWhiteSpace(registrationFailure))
            {
                bool missing = registrationFailure.IndexOf("not found", StringComparison.OrdinalIgnoreCase) >= 0 ||
                               registrationFailure.IndexOf("missing", StringComparison.OrdinalIgnoreCase) >= 0;
                bool loadFailure = registrationFailure.IndexOf("failed to preload", StringComparison.OrdinalIgnoreCase) >= 0 ||
                                   registrationFailure.IndexOf("failed to load", StringComparison.OrdinalIgnoreCase) >= 0;
                OnnxProviderReadinessKind kind = missing
                    ? OnnxProviderReadinessKind.MissingPayload
                    : loadFailure ? OnnxProviderReadinessKind.MissingNativeDependencies : OnnxProviderReadinessKind.RegistrationFailed;
                return State(kind,
                    missing ? "Payload missing" : loadFailure ? "Driver/runtime missing" : "Registration failed",
                    registrationFailure);
            }
            if (!runtimeAvailable)
                return State(OnnxProviderReadinessKind.UnavailableInRuntime, "Unavailable in runtime", "The loaded ONNX Runtime does not advertise this execution provider.");
            if (!probeComplete)
                return State(OnnxProviderReadinessKind.Checking, "Checking...", "Native provider initialization is being checked.");
            if (!string.IsNullOrWhiteSpace(probeFailure))
            {
                bool invalidDevice = IsInvalidDeviceFailure(probeFailure);
                bool unsupportedHardware = IsUnsupportedHardwareFailure(probeFailure);
                return State(
                    invalidDevice ? OnnxProviderReadinessKind.InvalidDevice : unsupportedHardware
                        ? OnnxProviderReadinessKind.UnsupportedHardware
                        : OnnxProviderReadinessKind.MissingNativeDependencies,
                    invalidDevice ? "Invalid device" : unsupportedHardware ? "Unsupported GPU" : "Driver/runtime missing",
                    probeFailure);
            }
            return State(OnnxProviderReadinessKind.Ready, "Ready", "The provider is included, advertised by ONNX Runtime, and initialized successfully.");
        }

        internal static OnnxProviderReadiness EvaluatePlayer(
            OnnxExecutionProvider provider,
            bool configured,
            RuntimePlatform? platform,
            bool included,
            bool payloadPresent,
            bool hasActiveProfile)
        {
            if (!configured)
                return State(OnnxProviderReadinessKind.NotConfigured, "Not configured", "This provider is not present in Provider Order.");
            if (!platform.HasValue || !OnnxRuntimeBackend.IsProviderValidForPlatform(provider, platform.Value))
                return State(OnnxProviderReadinessKind.Unsupported, "Unsupported", platform.HasValue
                    ? $"{DisplayName(provider)} is not supported on {platform.Value}."
                    : "The active build target is not supported by this backend.");
            if (!included)
            {
                string reason = provider == OnnxExecutionProvider.Cpu
                    ? "The ONNX Runtime core is missing for the active build target."
                    : hasActiveProfile
                        ? $"The active Build Profile does not include {OnnxRuntimeProviderPluginFilter.GetDefine(provider)}."
                        : "No Build Profile is active; optional provider defines are absent and only CPU is packaged.";
                return State(OnnxProviderReadinessKind.NotIncluded, "Not included", reason);
            }
            if (!payloadPresent)
            {
                return State(
                    OnnxProviderReadinessKind.MissingPayload,
                    "Payload missing",
                    "The required native provider payload is not present in its owning package.");
            }
            return State(OnnxProviderReadinessKind.Ready, "Included", "The active Player build will include this provider. Device readiness is verified only at runtime.");
        }

        internal static string DisplayName(OnnxExecutionProvider provider)
        {
            return provider switch
            {
                OnnxExecutionProvider.TensorRt => "TensorRT",
                OnnxExecutionProvider.TensorRtRtx => "TensorRT-RTX",
                OnnxExecutionProvider.DirectMl => "DirectML",
                OnnxExecutionProvider.WebGpu => "WebGPU",
                OnnxExecutionProvider.CoreMl => "CoreML",
                OnnxExecutionProvider.OpenVino => "OpenVINO",
                OnnxExecutionProvider.Nnapi => "NNAPI",
                OnnxExecutionProvider.Cuda => "CUDA",
                _ => "CPU"
            };
        }

        internal static string RuntimeName(OnnxExecutionProvider provider)
        {
            return provider switch
            {
                OnnxExecutionProvider.TensorRt => "TensorrtExecutionProvider",
                OnnxExecutionProvider.TensorRtRtx => "NvTensorRTRTXExecutionProvider",
                OnnxExecutionProvider.DirectMl => "DmlExecutionProvider",
                OnnxExecutionProvider.WebGpu => "WebGpuExecutionProvider",
                OnnxExecutionProvider.CoreMl => "CoreMLExecutionProvider",
                OnnxExecutionProvider.OpenVino => "OpenVINOExecutionProvider",
                OnnxExecutionProvider.Nnapi => "NnapiExecutionProvider",
                OnnxExecutionProvider.Cuda => "CUDAExecutionProvider",
                _ => "CPUExecutionProvider"
            };
        }

        internal static string Dependencies(OnnxExecutionProvider provider)
        {
            return provider switch
            {
                OnnxExecutionProvider.TensorRt => "Compatible NVIDIA driver, CUDA, cuDNN, and TensorRT native libraries.",
                OnnxExecutionProvider.TensorRtRtx => "RTX 30-series/Ampere or newer GPU and a compatible NVIDIA driver. Runtime libraries are bundled by the NVIDIA package.",
                OnnxExecutionProvider.Cuda => "Compatible NVIDIA driver. CUDA 12 and cuDNN 9 runtime libraries are bundled on Windows and Linux.",
                OnnxExecutionProvider.DirectMl => "Windows x64 with a DirectX 12-capable GPU driver.",
                OnnxExecutionProvider.WebGpu => "Linux x64 with a Vulkan-capable GPU driver.",
                OnnxExecutionProvider.CoreMl => "Apple-silicon macOS with CoreML support.",
                OnnxExecutionProvider.OpenVino => "Compatible OpenVINO runtime libraries on Linux x64.",
                OnnxExecutionProvider.Nnapi => "Android device API level 27 or newer.",
                _ => "No external execution-provider dependencies."
            };
        }

        internal static string DependencyStatus(OnnxExecutionProvider provider, bool payloadPresent)
        {
            string[] required = provider switch
            {
                OnnxExecutionProvider.Cuda => new[]
                {
                    "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll", "cudart64_12.dll",
                    "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "curand64_10.dll", "cudnn64_9.dll"
                },
                OnnxExecutionProvider.TensorRtRtx => new[]
                {
                    "onnxruntime_providers_nv_tensorrt_rtx.dll", "cudart64_12.dll", "nvrtc64_120_0.dll",
                    "nvJitLink_120_0.dll", "tensorrt_rtx_1_5.dll", "tensorrt_onnxparser_rtx_1_5.dll"
                },
                _ => Array.Empty<string>()
            };
            if (required.Length == 0 || Application.platform != RuntimePlatform.WindowsEditor)
                return Dependencies(provider);

            if (provider == OnnxExecutionProvider.Cuda && required.All(IsBundledWindowsLibrary))
                return "CUDA 12 runtime and cuDNN 9 are bundled with this package. A compatible NVIDIA display driver is still required.";

            string[] missing = required.Where(library => !CanResolveNativeLibrary(library)).ToArray();
            if (missing.Length == 0)
                return $"{Dependencies(provider)} All direct provider DLL imports are discoverable; an initialization failure now points to an ABI/version mismatch or a transitive dependency.";
            return $"Missing from the Editor process DLL search path: {string.Join(", ", missing)}. " +
                   $"Add their directories to {OnnxRuntimeBackend.NativeDependencyPathEnvironmentVariable} before the first ONNX Runtime use.";
        }

        private static bool CanResolveNativeLibrary(string fileName)
        {
            if (IsBundledWindowsLibrary(fileName)) return true;
            IEnumerable<string> directories = new[] { AppContext.BaseDirectory, Environment.CurrentDirectory }
                .Concat((Environment.GetEnvironmentVariable("PATH") ?? string.Empty)
                    .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries));
            return directories.Any(directory =>
            {
                try { return File.Exists(Path.Combine(directory.Trim().Trim('"'), fileName)); }
                catch { return false; }
            });
        }

        private static bool IsBundledWindowsLibrary(string fileName)
        {
            const string root = "Packages/ai.kitsumate.onnx.backend.onnxruntime.nvidia/Runtime/Plugins/Windows/x86_64";
            return AssetDatabase.FindAssets(Path.GetFileNameWithoutExtension(fileName), new[] { root })
                .Select(AssetDatabase.GUIDToAssetPath)
                .Any(path => string.Equals(Path.GetFileName(path), fileName, StringComparison.OrdinalIgnoreCase));
        }

        private static OnnxProviderReadiness State(OnnxProviderReadinessKind kind, string text, string reason) => new(kind, text, reason);

        private static bool IsInvalidDeviceFailure(string failure)
        {
            return failure.IndexOf("invalid device", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   failure.IndexOf("device ordinal", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   failure.IndexOf("device id", StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private static bool IsUnsupportedHardwareFailure(string failure)
        {
            return failure.IndexOf("unsupported gpu", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   failure.IndexOf("unsupported device", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   failure.IndexOf("compute capability", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   failure.IndexOf("ampere", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   failure.IndexOf("no compatible", StringComparison.OrdinalIgnoreCase) >= 0;
        }
    }

    internal sealed class OnnxProviderProbeSnapshot
    {
        internal bool Complete;
        internal string RuntimeVersion = string.Empty;
        internal string RuntimeFailure;
        internal readonly HashSet<string> AvailableProviders = new(StringComparer.OrdinalIgnoreCase);
        internal readonly Dictionary<OnnxExecutionProvider, string> RegistrationFailures = new();
        internal readonly Dictionary<OnnxExecutionProvider, string> Failures = new();
        internal readonly Dictionary<OnnxExecutionProvider, IReadOnlyList<OnnxExecutionDeviceInfo>> Devices = new();
        internal readonly Dictionary<OnnxExecutionProvider, OnnxExecutionDeviceInfo> SelectedDevices = new();
        internal readonly Dictionary<OnnxExecutionProvider, string> DeviceSelectionReasons = new();
    }

    [InitializeOnLoad]
    internal static class OnnxProviderProbeCache
    {
        private static readonly Dictionary<string, OnnxProviderProbeSnapshot> Entries = new();

        static OnnxProviderProbeCache()
        {
            AssemblyReloadEvents.beforeAssemblyReload += Invalidate;
        }

        internal static OnnxProviderProbeSnapshot GetOrStart(
            string key,
            RuntimePlatform platform,
            OnnxDeviceSelectionMode selectionMode,
            int deviceId)
        {
            if (Entries.TryGetValue(key, out OnnxProviderProbeSnapshot existing)) return existing;
            var pending = new OnnxProviderProbeSnapshot();
            Entries[key] = pending;
            RunAsync(key, platform, selectionMode, deviceId);
            return pending;
        }

        internal static void Invalidate() => Entries.Clear();

        private static async void RunAsync(
            string key,
            RuntimePlatform platform,
            OnnxDeviceSelectionMode selectionMode,
            int deviceId)
        {
            OnnxProviderProbeSnapshot result = await Task.Run(() => Probe(platform, selectionMode, deviceId));
            if (!Entries.ContainsKey(key)) return;
            Entries[key] = result;
            InternalEditorUtility.RepaintAllViews();
        }

        private static OnnxProviderProbeSnapshot Probe(
            RuntimePlatform platform,
            OnnxDeviceSelectionMode selectionMode,
            int deviceId)
        {
            var result = new OnnxProviderProbeSnapshot();
            try
            {
                result.RuntimeVersion = OnnxRuntimeBackend.GetRuntimeVersion();
                foreach (IOnnxRuntimeProviderModule module in OnnxRuntimeProviderRegistry.GetModules())
                {
                    if (!module.IsEnabled || !module.Supports(platform)) continue;
                    string failure = OnnxRuntimeBackend.ProbeProviderRegistration(module.Provider);
                    if (!string.IsNullOrWhiteSpace(failure)) result.RegistrationFailures[module.Provider] = failure;
                }
                foreach (string provider in OnnxRuntimeBackend.GetRuntimeAvailableProviderNames())
                    result.AvailableProviders.Add(provider);
                foreach (OnnxExecutionProvider provider in Enum.GetValues(typeof(OnnxExecutionProvider)))
                {
                    if (!OnnxRuntimeBackend.IsProviderValidForPlatform(provider, platform)) continue;
                    if (result.RegistrationFailures.ContainsKey(provider)) continue;
                    if (!result.AvailableProviders.Contains(OnnxProviderReadinessEvaluator.RuntimeName(provider))) continue;
                    int resolvedDeviceId = deviceId;
                    try
                    {
                        if (OnnxRuntimeProviderRegistry.TryGet(provider, out IOnnxRuntimeProviderModule module))
                            result.Devices[provider] = module.GetDevices();
                        OnnxExecutionDeviceInfo selected = OnnxRuntimeBackend.ResolveDevice(
                            provider, selectionMode, deviceId, out string selectionReason);
                        result.SelectedDevices[provider] = selected;
                        result.DeviceSelectionReasons[provider] = selectionReason;
                        resolvedDeviceId = selected.ProviderDeviceIndex;
                    }
                    catch (Exception exception)
                    {
                        result.Failures[provider] = exception.GetBaseException().Message;
                        continue;
                    }
                    string failure = OnnxRuntimeBackend.ProbeProviderInitialization(provider, resolvedDeviceId);
                    if (!string.IsNullOrWhiteSpace(failure)) result.Failures[provider] = failure;
                }
            }
            catch (Exception exception)
            {
                Exception root = exception.GetBaseException();
                result.RuntimeFailure = $"{root.GetType().Name}: {root.Message}";
            }
            result.Complete = true;
            return result;
        }
    }

    internal sealed class OnnxRuntimeProviderAssetPostprocessor : AssetPostprocessor
    {
        private static void OnPostprocessAllAssets(
            string[] importedAssets,
            string[] deletedAssets,
            string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            if (importedAssets.Concat(deletedAssets).Concat(movedAssets).Concat(movedFromAssetPaths)
                .Any(path => path.Contains("ai.kitsumate.onnx.backend.onnxruntime", StringComparison.OrdinalIgnoreCase)))
            {
                OnnxProviderProbeCache.Invalidate();
                OnnxProviderPayloadCache.Invalidate();
            }
        }
    }

    internal static class OnnxProviderPayloadCache
    {
        private static readonly Dictionary<(BuildTarget Target, OnnxExecutionProvider Provider), bool> Entries = new();

        internal static bool Has(BuildTarget target, OnnxExecutionProvider provider)
        {
            var key = (target, provider);
            if (Entries.TryGetValue(key, out bool present)) return present;
            present = Find(target, provider);
            Entries[key] = present;
            return present;
        }

        internal static void Invalidate() => Entries.Clear();

        private static bool Find(BuildTarget target, OnnxExecutionProvider provider)
        {
            string platformFolder = target switch
            {
                BuildTarget.StandaloneWindows64 => "Windows/x86_64",
                BuildTarget.StandaloneLinux64 => "Linux/x86_64",
                BuildTarget.StandaloneOSX => "macOS/arm64",
                BuildTarget.Android => "Android",
                _ => null
            };
            if (platformFolder == null) return false;
            string token = provider switch
            {
                OnnxExecutionProvider.Cuda => "cuda",
                OnnxExecutionProvider.TensorRtRtx => "nv_tensorrt_rtx",
                OnnxExecutionProvider.DirectMl => "directml",
                OnnxExecutionProvider.WebGpu => "webgpu",
                OnnxExecutionProvider.CoreMl => "onnxruntime",
                OnnxExecutionProvider.OpenVino => "openvino",
                _ => "onnxruntime"
            };
            string package = provider is OnnxExecutionProvider.Cuda or OnnxExecutionProvider.TensorRtRtx
                ? "ai.kitsumate.onnx.backend.onnxruntime.nvidia"
                : "ai.kitsumate.onnx.backend.onnxruntime";
            string root = $"Packages/{package}/Runtime/Plugins/{platformFolder}";
            return AssetDatabase.FindAssets(token, new[] { root })
                .Select(AssetDatabase.GUIDToAssetPath)
                .Any(path => System.IO.Path.GetFileName(path).Contains(token, StringComparison.OrdinalIgnoreCase));
        }
    }

    [CustomEditor(typeof(OnnxRuntimeBackend))]
    public sealed class OnnxRuntimeBackendEditor : UnityEditor.Editor
    {
        private const float ProviderWidth = 92f;
        private readonly Dictionary<OnnxExecutionProvider, bool> _expanded = new();
        private SerializedProperty _providerSelectionMode;
        private SerializedProperty _providerOrder;
        private SerializedProperty _gpuDeviceId;
        private SerializedProperty _deviceSelectionMode;
        private ReorderableList _providerList;

        private void OnEnable()
        {
            _providerSelectionMode = serializedObject.FindProperty("_providerSelectionMode");
            _providerOrder = serializedObject.FindProperty("_providerOrder");
            _gpuDeviceId = serializedObject.FindProperty("_gpuDeviceId");
            _deviceSelectionMode = serializedObject.FindProperty("_deviceSelectionMode");
            _providerList = new ReorderableList(serializedObject, _providerOrder, true, true, true, true)
            {
                drawHeaderCallback = rect => EditorGUI.LabelField(rect, "Provider Order (first available wins)"),
                drawElementCallback = DrawProviderElement,
                onCanRemoveCallback = list => list.count > 1,
                onAddDropdownCallback = ShowAddProviderMenu
            };
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            var backend = (OnnxRuntimeBackend)target;
            IReadOnlyList<OnnxExecutionProvider> configured =
                (OnnxProviderSelectionMode)_providerSelectionMode.enumValueIndex == OnnxProviderSelectionMode.Automatic
                    ? backend.ProviderOrder
                    : SerializedProviderOrder();
            RuntimePlatform editorPlatform = Application.platform;
            BuildTarget playerTarget = EditorUserBuildSettings.activeBuildTarget;
            RuntimePlatform? playerPlatform = ToRuntimePlatform(playerTarget);
            BuildProfile profile = BuildProfile.GetActiveBuildProfile();
            HashSet<string> defines = OnnxRuntimeProviderPluginFilter.GetActiveDefines();
            IReadOnlyList<OnnxExecutionProvider> editorPackaged = OnnxRuntimeBackend.GetPackagedProviders();
            IReadOnlyList<OnnxExecutionProvider> playerPackaged = OnnxRuntimeProviderPluginFilter.GetPackagedProviders(playerTarget, defines);
            string profileId = profile == null ? "0" : profile.GetEntityId().ToString();
            string runtimeVersion = OnnxRuntimeBackend.GetRuntimeVersion();
            var deviceMode = (OnnxDeviceSelectionMode)_deviceSelectionMode.enumValueIndex;
            string cacheKey = $"{editorPlatform}|{deviceMode}|{_gpuDeviceId.intValue}|{profileId}|{runtimeVersion}|{string.Join(",", defines.OrderBy(value => value))}";
            OnnxProviderProbeSnapshot probe = OnnxProviderProbeCache.GetOrStart(
                cacheKey, editorPlatform, deviceMode, _gpuDeviceId.intValue);

            DrawEnvironment(editorPlatform, playerTarget, profile, configured, editorPackaged, probe);
            EditorGUILayout.Space(4);
            DrawReadinessTable(configured, editorPlatform, playerPlatform, profile != null, editorPackaged, playerPackaged, probe, playerTarget);
            EditorGUILayout.Space(4);
            EditorGUILayout.HelpBox("Ready verifies provider initialization only. Model compatibility is checked when a session is created. Player inclusion does not guarantee device readiness.", MessageType.Info);
            EditorGUILayout.Space(4);

            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(_providerSelectionMode, new GUIContent("Provider Selection"));
            using (new EditorGUI.DisabledScope(
                       (OnnxProviderSelectionMode)_providerSelectionMode.enumValueIndex == OnnxProviderSelectionMode.Automatic))
                _providerList.DoLayoutList();
            EditorGUILayout.PropertyField(_deviceSelectionMode, new GUIContent("Device Selection"));
            using (new EditorGUI.DisabledScope(deviceMode == OnnxDeviceSelectionMode.Automatic))
                EditorGUILayout.PropertyField(_gpuDeviceId, new GUIContent("Provider Device Index"));
            if (deviceMode == OnnxDeviceSelectionMode.Automatic)
                EditorGUILayout.HelpBox(
                    "Each provider independently matches Unity's active render GPU. If no exact PCI ID match exists, a single accelerator is selected, otherwise provider device 0 is used with a diagnostic warning.",
                    MessageType.Info);
            bool changed = EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
            if (changed) EditorUtility.SetDirty(backend);

            if (configured.Count == 0)
                EditorGUILayout.HelpBox("Provider Order must contain at least one provider.", MessageType.Error);
            else if (configured.Distinct().Count() != configured.Count)
                EditorGUILayout.HelpBox("Provider Order contains duplicates. Each provider can appear only once.", MessageType.Error);
        }

        private void DrawEnvironment(
            RuntimePlatform editorPlatform,
            BuildTarget playerTarget,
            BuildProfile profile,
            IReadOnlyList<OnnxExecutionProvider> configured,
            IReadOnlyList<OnnxExecutionProvider> editorPackaged,
            OnnxProviderProbeSnapshot probe)
        {
            EditorGUILayout.LabelField("Environment", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Editor", FormatPlatform(editorPlatform));
            EditorGUILayout.LabelField("Player", FormatBuildTarget(playerTarget));
            EditorGUILayout.LabelField("Build Profile", profile == null ? "None (optional providers disabled)" : profile.name);
            string expected = "None";
            if (!probe.Complete) expected = "Checking...";
            else
            {
                foreach (OnnxExecutionProvider provider in configured)
                {
                    bool advertised = probe.AvailableProviders.Contains(OnnxProviderReadinessEvaluator.RuntimeName(provider));
                    probe.RegistrationFailures.TryGetValue(provider, out string registrationFailure);
                    probe.Failures.TryGetValue(provider, out string failure);
                    OnnxProviderReadiness readiness = OnnxProviderReadinessEvaluator.EvaluateEditor(
                        provider, true, editorPlatform, editorPackaged.Contains(provider), advertised,
                        probe.Complete, failure, registrationFailure, probe.RuntimeFailure);
                    if (readiness.Kind != OnnxProviderReadinessKind.Ready) continue;
                    expected = OnnxProviderReadinessEvaluator.DisplayName(provider);
                    break;
                }
            }
            EditorGUILayout.LabelField("Expected Editor Provider", expected);
        }

        private void DrawReadinessTable(
            IReadOnlyList<OnnxExecutionProvider> configured,
            RuntimePlatform editorPlatform,
            RuntimePlatform? playerPlatform,
            bool hasActiveProfile,
            IReadOnlyList<OnnxExecutionProvider> editorPackaged,
            IReadOnlyList<OnnxExecutionProvider> playerPackaged,
            OnnxProviderProbeSnapshot probe,
            BuildTarget playerTarget)
        {
            EditorGUILayout.LabelField("Execution Provider Readiness", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Label("Provider", EditorStyles.miniBoldLabel, GUILayout.Width(ProviderWidth));
                GUILayout.Label("Editor", EditorStyles.miniBoldLabel, GUILayout.MinWidth(120));
                GUILayout.Label($"{FormatBuildTarget(playerTarget)} Player", EditorStyles.miniBoldLabel, GUILayout.MinWidth(120));
            }

            foreach (OnnxExecutionProvider provider in Enum.GetValues(typeof(OnnxExecutionProvider)).Cast<OnnxExecutionProvider>().OrderBy(DisplayOrder))
            {
                bool isConfigured = configured.Contains(provider);
                bool advertised = probe.AvailableProviders.Contains(OnnxProviderReadinessEvaluator.RuntimeName(provider));
                probe.RegistrationFailures.TryGetValue(provider, out string registrationFailure);
                probe.Failures.TryGetValue(provider, out string probeFailure);
                OnnxProviderReadiness editor = OnnxProviderReadinessEvaluator.EvaluateEditor(
                    provider, isConfigured, editorPlatform, editorPackaged.Contains(provider), advertised,
                    probe.Complete, probeFailure, registrationFailure, probe.RuntimeFailure);
                bool payload = OnnxProviderPayloadCache.Has(playerTarget, provider);
                OnnxProviderReadiness player = OnnxProviderReadinessEvaluator.EvaluatePlayer(
                    provider, isConfigured, playerPlatform, playerPackaged.Contains(provider), payload, hasActiveProfile);

                using (new EditorGUILayout.HorizontalScope(EditorStyles.helpBox))
                {
                    bool expanded = _expanded.TryGetValue(provider, out bool value) && value;
                    Rect foldoutRect = GUILayoutUtility.GetRect(
                        ProviderWidth,
                        EditorGUIUtility.singleLineHeight,
                        GUILayout.Width(ProviderWidth));
                    _expanded[provider] = EditorGUI.Foldout(
                        foldoutRect,
                        expanded,
                        OnnxProviderReadinessEvaluator.DisplayName(provider),
                        true);
                    DrawStatus(editor);
                    DrawStatus(player);
                }
                if (_expanded[provider]) DrawDetails(provider, configured, editorPlatform, playerPlatform, hasActiveProfile, editorPackaged, playerPackaged, advertised, probe, probeFailure, registrationFailure, payload, editor, player);
            }
        }

        private static void DrawStatus(OnnxProviderReadiness readiness)
        {
            GUIContent icon = EditorGUIUtility.IconContent(readiness.Kind switch
            {
                OnnxProviderReadinessKind.Ready => "TestPassed",
                OnnxProviderReadinessKind.Checking => "console.infoicon.sml",
                OnnxProviderReadinessKind.NotConfigured => "console.infoicon.sml",
                OnnxProviderReadinessKind.Unsupported => "console.warnicon.sml",
                OnnxProviderReadinessKind.NotIncluded => "console.warnicon.sml",
                _ => "console.erroricon.sml"
            });
            using (new EditorGUILayout.HorizontalScope(GUILayout.MinWidth(120)))
            {
                GUILayout.Label(icon, GUILayout.Width(18));
                GUILayout.Label(new GUIContent(readiness.Text, readiness.Reason), EditorStyles.miniLabel);
            }
        }

        private static void DrawDetails(
            OnnxExecutionProvider provider,
            IReadOnlyList<OnnxExecutionProvider> configured,
            RuntimePlatform editorPlatform,
            RuntimePlatform? playerPlatform,
            bool hasActiveProfile,
            IReadOnlyList<OnnxExecutionProvider> editorPackaged,
            IReadOnlyList<OnnxExecutionProvider> playerPackaged,
            bool advertised,
            OnnxProviderProbeSnapshot probe,
            string probeFailure,
            string registrationFailure,
            bool payload,
            OnnxProviderReadiness editor,
            OnnxProviderReadiness player)
        {
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                int index = configured.ToList().IndexOf(provider);
                Detail("Configuration", index >= 0 ? $"Fallback #{index + 1}" : "Not present in Provider Order");
                Detail("Editor platform", Gate(OnnxRuntimeBackend.IsProviderValidForPlatform(provider, editorPlatform), editorPlatform.ToString()));
                Detail("Editor inclusion", Gate(editorPackaged.Contains(provider), editorPackaged.Contains(provider) ? "Compiled into this Editor" : "Disabled by compilation defines"));
                Detail("ORT runtime", string.IsNullOrWhiteSpace(probe.RuntimeFailure)
                    ? Gate(advertised, advertised ? $"Advertised ({OnnxProviderReadinessEvaluator.RuntimeName(provider)})" : "Not advertised")
                    : $"Failed: {probe.RuntimeFailure}");
                Detail("Plugin registration", string.IsNullOrWhiteSpace(registrationFailure) ? "Passed or built in" : $"Failed: {registrationFailure}");
                Detail("Native check", !probe.Complete ? "Checking..." : !advertised ? "Not attempted" : string.IsNullOrWhiteSpace(probeFailure) ? "Passed" : $"Failed: {probeFailure}");
                if (probe.SelectedDevices.TryGetValue(provider, out OnnxExecutionDeviceInfo selectedDevice))
                {
                    probe.DeviceSelectionReasons.TryGetValue(provider, out string selectionReason);
                    Detail("Editor device", $"{selectedDevice.DisplayName} — {selectionReason}");
                }
                if (probe.Devices.TryGetValue(provider, out IReadOnlyList<OnnxExecutionDeviceInfo> devices) && devices.Count > 1)
                    Detail("Available devices", string.Join("; ", devices.Select(device => device.DisplayName)));
                Detail("Player platform", playerPlatform.HasValue ? Gate(OnnxRuntimeBackend.IsProviderValidForPlatform(provider, playerPlatform.Value), playerPlatform.Value.ToString()) : "Unsupported build target");
                Detail("Build Profile", hasActiveProfile ? (OnnxRuntimeProviderPluginFilter.GetDefine(provider) ?? "CPU is always included") : "None; optional provider defines are absent");
                Detail("Player packaging", Gate(playerPackaged.Contains(provider), playerPackaged.Contains(provider) ? "Included by profile" : "Not included by profile"));
                Detail("Native payload", Gate(payload, payload ? "Present in package" : "Missing from package"));
                Detail("Dependencies", OnnxProviderReadinessEvaluator.DependencyStatus(provider, payload));
                if (editor.Kind != OnnxProviderReadinessKind.Ready) Detail("Editor reason", editor.Reason);
                if (player.Kind != OnnxProviderReadinessKind.Ready) Detail("Player reason", player.Reason);
            }
        }

        private void DrawProviderElement(Rect rect, int index, bool active, bool focused)
        {
            SerializedProperty element = _providerOrder.GetArrayElementAtIndex(index);
            EditorGUI.PropertyField(rect, element, GUIContent.none);
        }

        private void ShowAddProviderMenu(Rect buttonRect, ReorderableList list)
        {
            IReadOnlyList<OnnxExecutionProvider> configured = SerializedProviderOrder();
            var menu = new GenericMenu();
            foreach (OnnxExecutionProvider provider in Enum.GetValues(typeof(OnnxExecutionProvider)))
            {
                OnnxExecutionProvider captured = provider;
                if (configured.Contains(provider)) menu.AddDisabledItem(new GUIContent(OnnxProviderReadinessEvaluator.DisplayName(provider)));
                else menu.AddItem(new GUIContent(OnnxProviderReadinessEvaluator.DisplayName(provider)), false, () =>
                {
                    serializedObject.Update();
                    int index = _providerOrder.arraySize++;
                    _providerOrder.GetArrayElementAtIndex(index).enumValueIndex = (int)captured;
                    serializedObject.ApplyModifiedProperties();
                });
            }
            menu.DropDown(buttonRect);
        }

        private IReadOnlyList<OnnxExecutionProvider> SerializedProviderOrder()
        {
            var values = new List<OnnxExecutionProvider>(_providerOrder.arraySize);
            for (int i = 0; i < _providerOrder.arraySize; i++)
                values.Add((OnnxExecutionProvider)_providerOrder.GetArrayElementAtIndex(i).enumValueIndex);
            return values;
        }

        private static RuntimePlatform? ToRuntimePlatform(BuildTarget target)
        {
            return target switch
            {
                BuildTarget.StandaloneWindows64 => RuntimePlatform.WindowsPlayer,
                BuildTarget.StandaloneLinux64 => RuntimePlatform.LinuxPlayer,
                BuildTarget.StandaloneOSX => RuntimePlatform.OSXPlayer,
                BuildTarget.Android => RuntimePlatform.Android,
                _ => null
            };
        }

        private static string FormatPlatform(RuntimePlatform platform)
        {
            return platform switch
            {
                RuntimePlatform.WindowsEditor => "Windows Editor",
                RuntimePlatform.WindowsPlayer => "Windows Player",
                RuntimePlatform.LinuxEditor => "Linux Editor",
                RuntimePlatform.LinuxPlayer => "Linux Player",
                RuntimePlatform.OSXEditor => "macOS Editor",
                RuntimePlatform.OSXPlayer => "macOS Player",
                RuntimePlatform.Android => "Android",
                _ => platform.ToString()
            };
        }

        private static string FormatBuildTarget(BuildTarget target)
        {
            return target switch
            {
                BuildTarget.StandaloneWindows64 => "Windows",
                BuildTarget.StandaloneLinux64 => "Linux",
                BuildTarget.StandaloneOSX => "macOS",
                BuildTarget.Android => "Android",
                _ => target.ToString()
            };
        }

        private static int DisplayOrder(OnnxExecutionProvider provider)
        {
            return provider switch
            {
                OnnxExecutionProvider.TensorRtRtx => 0,
                OnnxExecutionProvider.Cuda => 1,
                OnnxExecutionProvider.DirectMl or OnnxExecutionProvider.WebGpu or OnnxExecutionProvider.CoreMl or OnnxExecutionProvider.Nnapi => 2,
                OnnxExecutionProvider.Cpu => 3,
                _ => 4
            };
        }

        private static string Gate(bool passed, string text) => $"{(passed ? "Pass" : "Fail")}: {text}";
        private static void Detail(string label, string value) => EditorGUILayout.LabelField(label, value, EditorStyles.wordWrappedMiniLabel);
    }
}
#endif

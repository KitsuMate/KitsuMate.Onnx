using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using UnityEngine;

namespace KitsuMate.Onnx
{
    internal static class OnnxRuntimeGraphicsDeviceHint
    {
        private static readonly object Gate = new();
        private static uint _vendorId;
        private static uint _deviceId;
        private static string _name = string.Empty;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        internal static void CaptureFromUnity()
        {
            lock (Gate)
            {
                _vendorId = unchecked((uint)SystemInfo.graphicsDeviceVendorID);
                _deviceId = unchecked((uint)SystemInfo.graphicsDeviceID);
                _name = SystemInfo.graphicsDeviceName ?? string.Empty;
            }
        }

        internal static void Get(out uint vendorId, out uint deviceId, out string name)
        {
            lock (Gate)
            {
                vendorId = _vendorId;
                deviceId = _deviceId;
                name = _name;
            }
        }

        internal static void SetForTests(uint vendorId, uint deviceId, string name = null)
        {
            lock (Gate)
            {
                _vendorId = vendorId;
                _deviceId = deviceId;
                _name = name ?? string.Empty;
            }
        }
    }

    /// <summary>A separately deployable ONNX Runtime execution-provider integration.</summary>
    public interface IOnnxRuntimeProviderModule
    {
        OnnxExecutionProvider Provider { get; }
        string RuntimeProviderName { get; }
        int AutomaticPriority { get; }
        bool IsEnabled { get; }
        bool Supports(RuntimePlatform platform);
        void EnsureRegistered();
        IReadOnlyList<OnnxExecutionDeviceInfo> GetDevices();
        void Append(SessionOptions options, int deviceId);
    }

    /// <summary>
    /// Shared registry used by the default backend and extension packages.
    /// Registration is explicit so provider modules remain safe under Unity managed stripping.
    /// </summary>
    public static class OnnxRuntimeProviderRegistry
    {
        private static readonly object Gate = new();
        private static readonly Dictionary<OnnxExecutionProvider, IOnnxRuntimeProviderModule> Modules = new();
        private static readonly Dictionary<Assembly, HashSet<string>> NativeSearchRoots = new();

        static OnnxRuntimeProviderRegistry()
        {
            Register(new CpuProviderModule());
            Register(new DirectMlProviderModule());
            Register(new WebGpuProviderModule());
            Register(new CoreMlProviderModule());
            Register(new NnapiProviderModule());
        }

        public static void Register(IOnnxRuntimeProviderModule module)
        {
            if (module == null) throw new ArgumentNullException(nameof(module));
            lock (Gate)
            {
                if (Modules.TryGetValue(module.Provider, out IOnnxRuntimeProviderModule existing) &&
                    existing.GetType() != module.GetType())
                    throw new InvalidOperationException(
                        $"Provider '{module.Provider}' is already registered by {existing.GetType().FullName}.");
                Modules[module.Provider] = module;
            }
        }

        public static bool TryGet(
            OnnxExecutionProvider provider,
            out IOnnxRuntimeProviderModule module)
        {
            lock (Gate) return Modules.TryGetValue(provider, out module);
        }

        public static IReadOnlyList<IOnnxRuntimeProviderModule> GetModules()
        {
            lock (Gate) return Array.AsReadOnly(Modules.Values.ToArray());
        }

        /// <summary>
        /// Registers a resolved package/plugin root while running on Unity's main thread.
        /// Runtime provider discovery never calls AssetDatabase or PackageManager APIs.
        /// </summary>
        public static void RegisterNativeSearchRoot(Assembly moduleAssembly, string root)
        {
            if (moduleAssembly == null) throw new ArgumentNullException(nameof(moduleAssembly));
            if (string.IsNullOrWhiteSpace(root)) throw new ArgumentException("A native search root is required.", nameof(root));
            string fullPath = Path.GetFullPath(root);
            lock (Gate)
            {
                if (!NativeSearchRoots.TryGetValue(moduleAssembly, out HashSet<string> roots))
                    NativeSearchRoots[moduleAssembly] = roots = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                roots.Add(fullPath);
            }
        }

        internal static IReadOnlyList<string> GetNativeSearchRoots(Assembly moduleAssembly)
        {
            lock (Gate)
                return NativeSearchRoots.TryGetValue(moduleAssembly, out HashSet<string> roots)
                    ? roots.ToArray()
                    : Array.Empty<string>();
        }

        public static IReadOnlyList<OnnxExecutionProvider> ResolveAutomatic(RuntimePlatform platform)
        {
            lock (Gate)
            {
                return Array.AsReadOnly(Modules.Values
                    .Where(module => module.IsEnabled && module.Supports(platform))
                    .OrderByDescending(module => module.AutomaticPriority)
                    .ThenBy(module => (int)module.Provider)
                    .Select(module => module.Provider)
                    .ToArray());
            }
        }
    }

    /// <summary>Base for V2 plug-in EPs registered through OrtEnv.</summary>
    public abstract class OnnxRuntimePluginProviderModule : IOnnxRuntimeProviderModule
    {
        private readonly object _registrationGate = new();
        private bool _registered;
        private readonly List<IntPtr> _nativeDependencyHandles = new();

        public abstract OnnxExecutionProvider Provider { get; }
        public abstract string RuntimeProviderName { get; }
        public abstract int AutomaticPriority { get; }
        public abstract bool IsEnabled { get; }
        public abstract bool Supports(RuntimePlatform platform);
        protected abstract string RegistrationName { get; }
        protected abstract IReadOnlyList<string> LibraryNames { get; }
        protected virtual IReadOnlyList<string> DependencyLibraryNames => Array.Empty<string>();

        public IReadOnlyList<OnnxExecutionDeviceInfo> GetDevices()
        {
            if (!IsEnabled)
                throw new OnnxProviderUnavailableException(Provider, $"Provider module '{Provider}' is disabled by the active Build Profile.");
            OrtEnv env = OrtEnv.Instance();
            EnsureRegistered(env);
            return Array.AsReadOnly(env.GetEpDevices()
                .Where(device => string.Equals(device.EpName, RuntimeProviderName, StringComparison.OrdinalIgnoreCase))
                .Select((device, index) => ToDeviceInfo(Provider, device, index))
                .ToArray());
        }

        public void Append(SessionOptions options, int deviceId)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));
            OrtEnv env = OrtEnv.Instance();
            EnsureRegistered(env);
            OrtEpDevice[] devices = GetOrtDevices(env);
            if (devices.Length == 0)
                throw new OnnxProviderUnavailableException(
                    Provider,
                    $"The '{RuntimeProviderName}' library registered, but ONNX Runtime exposed no matching EP devices.");
            if ((uint)deviceId >= (uint)devices.Length)
                throw new OnnxProviderUnavailableException(
                    Provider,
                    $"Invalid device id {deviceId}; '{RuntimeProviderName}' exposed {devices.Length} device(s).");

            options.AppendExecutionProvider(
                env,
                new[] { devices[deviceId] },
                new Dictionary<string, string>());
        }

        public void EnsureRegistered() => EnsureRegistered(OrtEnv.Instance());

        protected virtual string LocateLibrary()
        {
            foreach (string libraryName in LibraryNames)
            {
                foreach (string directory in CandidateDirectories())
                {
                    string candidate = Path.Combine(directory, libraryName);
                    if (File.Exists(candidate)) return Path.GetFullPath(candidate);
                }
            }
            throw new DllNotFoundException(
                $"The native payload for '{Provider}' was not found. Expected one of: {string.Join(", ", LibraryNames)}.");
        }

        private void EnsureRegistered(OrtEnv env)
        {
            lock (_registrationGate)
            {
                if (_registered) return;
                string library = LocateLibrary();
                PreloadDependencies(Path.GetDirectoryName(library));
                env.RegisterExecutionProviderLibrary(RegistrationName, library);
                _registered = true;
            }
        }

        private void PreloadDependencies(string directory)
        {
            foreach (string name in DependencyLibraryNames)
            {
                string path = Path.Combine(directory ?? string.Empty, name);
                if (!File.Exists(path))
                    throw new DllNotFoundException($"Required native dependency for '{Provider}' is missing: {path}");
                IntPtr handle = Application.platform is RuntimePlatform.LinuxEditor or RuntimePlatform.LinuxPlayer
                    ? dlopen(path, 2 | 0x100)
                    : LoadLibraryEx(path, IntPtr.Zero, LoadLibrarySearchDllLoadDir | LoadLibrarySearchDefaultDirs);
                if (handle == IntPtr.Zero)
                    throw new DllNotFoundException(
                        $"Failed to preload native dependency for '{Provider}': {path} " +
                        $"(native error {Marshal.GetLastWin32Error()}).");
                _nativeDependencyHandles.Add(handle);
            }
        }

        private const uint LoadLibrarySearchDllLoadDir = 0x00000100;
        private const uint LoadLibrarySearchDefaultDirs = 0x00001000;

        [DllImport("kernel32", EntryPoint = "LoadLibraryExW", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr LoadLibraryEx(string path, IntPtr file, uint flags);

        [DllImport("libdl.so.2", SetLastError = true)]
        private static extern IntPtr dlopen(string path, int flags);

        protected OrtEpDevice[] GetOrtDevices(OrtEnv env) => env.GetEpDevices()
            .Where(device => string.Equals(device.EpName, RuntimeProviderName, StringComparison.OrdinalIgnoreCase))
            .ToArray();

        internal static OnnxExecutionDeviceInfo ToDeviceInfo(
            OnnxExecutionProvider provider, OrtEpDevice device, int index)
        {
            OrtHardwareDevice hardware = device.HardwareDevice;
            return new OnnxExecutionDeviceInfo(
                provider, index, hardware?.Type.ToString(), hardware?.VendorId ?? 0,
                hardware?.DeviceId ?? 0, hardware?.Vendor, device.EpVendor);
        }

        private IEnumerable<string> CandidateDirectories()
        {
            foreach (string root in OnnxRuntimeProviderRegistry.GetNativeSearchRoots(GetType().Assembly))
                foreach (string directory in EnumeratePluginDirectories(root))
                    yield return directory;
#if !UNITY_EDITOR
            // Unity deploys native plug-ins below the Player data directory. Search
            // the deployed layout rather than assuming a project/package path.
            foreach (string directory in EnumeratePluginDirectories(Path.Combine(Application.dataPath, "Plugins")))
                yield return directory;
#endif

            yield return AppContext.BaseDirectory;
            yield return Environment.CurrentDirectory;
            foreach (string directory in (Environment.GetEnvironmentVariable("PATH") ?? string.Empty)
                         .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
                yield return directory.Trim().Trim('"');
        }

        private static IEnumerable<string> EnumeratePluginDirectories(string root)
        {
            if (!Directory.Exists(root)) yield break;
            yield return root;
            foreach (string directory in Directory.EnumerateDirectories(root, "*", SearchOption.AllDirectories))
                yield return directory;
        }
    }

    internal sealed class CpuProviderModule : IOnnxRuntimeProviderModule
    {
        public OnnxExecutionProvider Provider => OnnxExecutionProvider.Cpu;
        public string RuntimeProviderName => "CPUExecutionProvider";
        public int AutomaticPriority => 0;
        public bool IsEnabled => true;
        public bool Supports(RuntimePlatform platform) => OnnxRuntimeBackend.IsSupportedPlatform(platform);
        public void EnsureRegistered() { }
        public IReadOnlyList<OnnxExecutionDeviceInfo> GetDevices() => new[]
        {
            new OnnxExecutionDeviceInfo(Provider, 0, "CPU", 0, 0, string.Empty, "Microsoft")
        };
        public void Append(SessionOptions options, int deviceId) => options.AppendExecutionProvider_CPU(0);
    }

    internal sealed class DirectMlProviderModule : IOnnxRuntimeProviderModule
    {
        public OnnxExecutionProvider Provider => OnnxExecutionProvider.DirectMl;
        public string RuntimeProviderName => "DmlExecutionProvider";
        public int AutomaticPriority => 300;
        public bool IsEnabled => true;
        public bool Supports(RuntimePlatform platform) =>
            platform is RuntimePlatform.WindowsEditor or RuntimePlatform.WindowsPlayer;
        public void EnsureRegistered() { }
        public IReadOnlyList<OnnxExecutionDeviceInfo> GetDevices()
        {
            OrtEpDevice[] devices = GetOrtDevices();
            return Array.AsReadOnly(devices.Select((device, index) =>
            {
                return OnnxRuntimePluginProviderModule.ToDeviceInfo(Provider, device, index);
            }).ToArray());
        }
        public void Append(SessionOptions options, int deviceId)
        {
            OrtEnv env = OrtEnv.Instance();
            OrtEpDevice[] devices = GetOrtDevices();
            if ((uint)deviceId >= (uint)devices.Length)
                throw new OnnxProviderUnavailableException(Provider,
                    $"Invalid device id {deviceId}; '{RuntimeProviderName}' exposed {devices.Length} device(s).");
            options.AppendExecutionProvider(env, new[] { devices[deviceId] }, new Dictionary<string, string>());
        }
        private OrtEpDevice[] GetOrtDevices() => OrtEnv.Instance().GetEpDevices()
            .Where(device => string.Equals(device.EpName, RuntimeProviderName, StringComparison.OrdinalIgnoreCase))
            .ToArray();
    }

    internal sealed class WebGpuProviderModule : OnnxRuntimePluginProviderModule
    {
        public override OnnxExecutionProvider Provider => OnnxExecutionProvider.WebGpu;
        public override string RuntimeProviderName => "WebGpuExecutionProvider";
        public override int AutomaticPriority => 300;
        public override bool IsEnabled => true;
        public override bool Supports(RuntimePlatform platform) =>
            platform is RuntimePlatform.LinuxEditor or RuntimePlatform.LinuxPlayer;
        protected override string RegistrationName => "kitsumate_webgpu";
        protected override IReadOnlyList<string> LibraryNames => new[] { "libonnxruntime_providers_webgpu.so" };
    }

    internal sealed class CoreMlProviderModule : IOnnxRuntimeProviderModule
    {
        public OnnxExecutionProvider Provider => OnnxExecutionProvider.CoreMl;
        public string RuntimeProviderName => "CoreMLExecutionProvider";
        public int AutomaticPriority => 300;
        public bool IsEnabled => true;
        public bool Supports(RuntimePlatform platform) =>
            platform is RuntimePlatform.OSXEditor or RuntimePlatform.OSXPlayer;
        public void EnsureRegistered() { }
        public IReadOnlyList<OnnxExecutionDeviceInfo> GetDevices() => new[]
        {
            new OnnxExecutionDeviceInfo(Provider, 0, "NPU", 0, 0, "Apple", "Apple")
        };
        public void Append(SessionOptions options, int deviceId) => options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
    }

    internal sealed class NnapiProviderModule : IOnnxRuntimeProviderModule
    {
        public OnnxExecutionProvider Provider => OnnxExecutionProvider.Nnapi;
        public string RuntimeProviderName => "NnapiExecutionProvider";
        public int AutomaticPriority => 300;
        public bool IsEnabled => true;
        public bool Supports(RuntimePlatform platform) => platform == RuntimePlatform.Android;
        public void EnsureRegistered() { }
        public IReadOnlyList<OnnxExecutionDeviceInfo> GetDevices() => new[]
        {
            new OnnxExecutionDeviceInfo(Provider, 0, "NPU", 0, 0, string.Empty, "Android")
        };
        public void Append(SessionOptions options, int deviceId)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            options.AppendExecutionProvider_Nnapi(NnapiFlags.NNAPI_FLAG_USE_FP16);
#else
            throw new PlatformNotSupportedException("NNAPI is only available in Android players.");
#endif
        }
    }
}

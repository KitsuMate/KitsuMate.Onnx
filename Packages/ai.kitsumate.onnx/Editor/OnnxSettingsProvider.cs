#if UNITY_EDITOR
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Project Settings provider for ONNX configuration.
    /// Access via Edit > Project Settings > KitsuMate > ONNX
    /// </summary>
    public class OnnxSettingsProvider : SettingsProvider
    {
        private SerializedObject _serializedSettings;
        private OnnxSettings _settings;
        
        private const string SettingsPath = "Project/KitsuMate/ONNX";
        
        public OnnxSettingsProvider(string path, SettingsScope scope)
            : base(path, scope)
        {
        }
        
        [SettingsProvider]
        public static SettingsProvider CreateSettingsProvider()
        {
            return new OnnxSettingsProvider(SettingsPath, SettingsScope.Project)
            {
                label = "ONNX",
                keywords = new HashSet<string>(new[] { "ONNX", "AI", "ML", "Inference", "Backend", "KitsuMate" })
            };
        }
        
        public override void OnActivate(string searchContext, VisualElement rootElement)
        {
            _settings = LoadOrCreateSettings();
            _serializedSettings = new SerializedObject(_settings);
        }
        
        public override void OnGUI(string searchContext)
        {
            if (_serializedSettings == null || _serializedSettings.targetObject == null)
            {
                _settings = LoadOrCreateSettings();
                _serializedSettings = new SerializedObject(_settings);
            }
            
            _serializedSettings.Update();
            
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("ONNX Settings", EditorStyles.boldLabel);
            EditorGUILayout.Space(5);
            
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Backend Configuration", EditorStyles.boldLabel);
                EditorGUILayout.Space(4);
                
                // Default Backend
                var defaultBackendProp = _serializedSettings.FindProperty("_defaultBackend");
                EditorGUILayout.PropertyField(defaultBackendProp, new GUIContent("Default Backend"));
                
                EditorGUILayout.Space(8);
                
                // Platform Overrides
                EditorGUILayout.LabelField("Platform Overrides", EditorStyles.boldLabel);
                var overridesProp = _serializedSettings.FindProperty("_platformOverrides");
                
                for (int i = 0; i < overridesProp.arraySize; i++)
                {
                    var element = overridesProp.GetArrayElementAtIndex(i);
                    var platformProp = element.FindPropertyRelative("Platform");
                    var backendProp = element.FindPropertyRelative("Backend");
                    
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        EditorGUILayout.PropertyField(platformProp, GUIContent.none, GUILayout.Width(150));
                        EditorGUILayout.PropertyField(backendProp, GUIContent.none);
                        
                        if (GUILayout.Button("×", GUILayout.Width(25)))
                        {
                            overridesProp.DeleteArrayElementAtIndex(i);
                            break;
                        }
                    }
                }
                
                EditorGUILayout.Space(4);
                if (GUILayout.Button("+ Add Platform Override"))
                {
                    overridesProp.InsertArrayElementAtIndex(overridesProp.arraySize);
                }
            }
            
            EditorGUILayout.Space(8);
            
            // Debug Options
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Debug Options", EditorStyles.boldLabel);
                var verboseProp = _serializedSettings.FindProperty("_verboseLogging");
                EditorGUILayout.PropertyField(verboseProp, new GUIContent("Verbose Logging"));
            }
            
            EditorGUILayout.Space(8);
            
            // Available Backends
            DrawAvailableBackends();
            
            EditorGUILayout.Space(8);
            
            // Current Platform Info
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Current Platform", EditorStyles.boldLabel);
                EditorGUILayout.LabelField("Platform", Application.platform.ToString());
                
                var currentBackend = _settings.GetBackendForCurrentPlatform();
                EditorGUILayout.LabelField("Active Backend", 
                    currentBackend != null ? currentBackend.DisplayName : "(none)");
            }
            
            _serializedSettings.ApplyModifiedProperties();
        }
        
        private void DrawAvailableBackends()
        {
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Available Backends", EditorStyles.boldLabel);
                
                var backends = FindAllBackends();
                if (backends.Length == 0)
                {
                    EditorGUILayout.HelpBox("No OnnxBackend assets found. Create one via Assets > Create > KitsuMate > ONNX > Backends.", MessageType.Info);
                }
                else
                {
                    foreach (var backend in backends)
                    {
                        using (new EditorGUILayout.HorizontalScope())
                        {
                            var icon = backend.IsAvailable 
                                ? EditorGUIUtility.IconContent("d_winbtn_mac_max").image 
                                : EditorGUIUtility.IconContent("d_winbtn_mac_close").image;
                            
                            GUILayout.Label(icon, GUILayout.Width(20), GUILayout.Height(16));
                            EditorGUILayout.LabelField(backend.DisplayName);
                            
                            EditorGUILayout.LabelField(
                                $"Priority: {backend.Priority}", 
                                EditorStyles.miniLabel, 
                                GUILayout.Width(80));
                            
                            if (GUILayout.Button("Select", EditorStyles.miniButton, GUILayout.Width(50)))
                            {
                                Selection.activeObject = backend;
                            }
                        }
                    }
                }
            }
        }
        
        private static OnnxBackend[] FindAllBackends()
        {
            return AssetDatabase.FindAssets("t:OnnxBackend")
                .Select(guid => AssetDatabase.LoadAssetAtPath<OnnxBackend>(AssetDatabase.GUIDToAssetPath(guid)))
                .Where(b => b != null)
                .OrderByDescending(b => b.Priority)
                .ToArray();
        }
        
        private static OnnxSettings LoadOrCreateSettings()
        {
            // Try to load existing settings
            var guids = AssetDatabase.FindAssets("t:OnnxSettings");
            if (guids.Length > 0)
            {
                var path = AssetDatabase.GUIDToAssetPath(guids[0]);
                return AssetDatabase.LoadAssetAtPath<OnnxSettings>(path);
            }
            
            // Create new settings
            var settings = ScriptableObject.CreateInstance<OnnxSettings>();
            
            // Ensure Resources folder exists
            const string resourcesPath = "Assets/Resources";
            if (!AssetDatabase.IsValidFolder(resourcesPath))
            {
                AssetDatabase.CreateFolder("Assets", "Resources");
            }
            
            const string settingsPath = resourcesPath + "/OnnxSettings.asset";
            AssetDatabase.CreateAsset(settings, settingsPath);
            AssetDatabase.SaveAssets();
            
            Debug.Log($"[KitsuMate.Onnx] Created OnnxSettings at {settingsPath}");
            return settings;
        }
    }
}
#endif

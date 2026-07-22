using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync
{
    /// <summary>
    /// Preset types for common character blend shape naming conventions.
    /// </summary>
    public enum VisemeProfilePreset
    {
        Custom,
        /// <summary>VRM 0.x / VRoid Studio: Fcl_MTH_A, Fcl_MTH_I, etc.</summary>
        VrmVroid,
        /// <summary>Daz3D Genesis 8/9: eCTRLvAA, eCTRLvBMP, etc.</summary>
        Daz3D,
        /// <summary>FACS visemes: matches *vAA* / *v_AA* style morphs.</summary>
        FACS,
        /// <summary>ARKit face tracking: jawOpen, mouthPucker, etc.</summary>
        ARKit,
        /// <summary>VRChat: vrc.v_sil, vrc.v_PP, vrc.v_AA, etc.</summary>
        VRChat
    }

    /// <summary>
    /// A single blend shape target with a name pattern and weight multiplier.
    /// The name may contain '*' wildcards for flexible matching.
    /// </summary>
    [Serializable]
    public struct BlendShapeTarget
    {
        [Tooltip("Blend shape name on the mesh. Supports '*' wildcards (e.g., '*_aa', 'Fcl_MTH_*').")]
        public string BlendShapeName;

        [Range(0f, 1f), Tooltip("Weight multiplier for this blend shape (0–1).")]
        public float Weight;

        public BlendShapeTarget(string blendShapeName, float weight = 1f)
        {
            BlendShapeName = blendShapeName;
            Weight = weight;
        }
    }

    /// <summary>
    /// Maps a single viseme to one or more blend shape targets.
    /// </summary>
    [Serializable]
    public struct VisemeBlendShapeMapping
    {
        public Viseme Viseme;
        public BlendShapeTarget[] Targets;

        public VisemeBlendShapeMapping(Viseme viseme, params BlendShapeTarget[] targets)
        {
            Viseme = viseme;
            Targets = targets;
        }
    }

    /// <summary>
    /// ScriptableObject that defines how visemes map to blend shapes on a character.
    /// Supports multiple blend shapes per viseme with individual weights, and wildcard name matching.
    /// Create presets for different character types (VRM, Daz3D, ARKit, custom).
    /// </summary>
    [CreateAssetMenu(fileName = "VisemeBlendShapeProfile", menuName = "KitsuMate/ONNX/LipSync/Viseme Blend Shape Profile")]
    public class VisemeBlendShapeProfile : ScriptableObject
    {
        [SerializeField]
        private VisemeBlendShapeMapping[] _mappings;

        /// <summary>All viseme-to-blend-shape mappings.</summary>
        public VisemeBlendShapeMapping[] Mappings
        {
            get => _mappings;
            set => _mappings = value;
        }

        /// <summary>
        /// Get the blend shape targets for a specific viseme.
        /// Returns an empty array if the viseme has no mapping.
        /// </summary>
        public BlendShapeTarget[] GetTargetsForViseme(Viseme viseme)
        {
            if (_mappings == null) return Array.Empty<BlendShapeTarget>();

            for (int i = 0; i < _mappings.Length; i++)
            {
                if (_mappings[i].Viseme == viseme)
                    return _mappings[i].Targets ?? Array.Empty<BlendShapeTarget>();
            }

            return Array.Empty<BlendShapeTarget>();
        }

        /// <summary>
        /// Populate mappings from a built-in preset.
        /// </summary>
        public void ApplyPreset(VisemeProfilePreset preset)
        {
            _mappings = preset switch
            {
                VisemeProfilePreset.VrmVroid => CreateVrmPreset(),
                VisemeProfilePreset.Daz3D => CreateDaz3DPreset(),
                VisemeProfilePreset.ARKit => CreateARKitPreset(),
                VisemeProfilePreset.FACS => CreateFacsPreset(),
                VisemeProfilePreset.VRChat => CreateVRChatPreset(),
                _ => CreateEmptyMappings()
            };
        }

        private static VisemeBlendShapeMapping[] CreateEmptyMappings()
        {
            var visemes = (Viseme[])Enum.GetValues(typeof(Viseme));
            var mappings = new VisemeBlendShapeMapping[visemes.Length];
            for (int i = 0; i < visemes.Length; i++)
            {
                mappings[i] = new VisemeBlendShapeMapping(visemes[i], Array.Empty<BlendShapeTarget>());
            }
            return mappings;
        }

        // ─────────────────────────────────────────────────────────────────
        //  VRM / VRoid preset (VRM 0.x Fcl_MTH_* naming, 5 mouth shapes)
        // ─────────────────────────────────────────────────────────────────
        private static VisemeBlendShapeMapping[] CreateVrmPreset()
        {
            return new[]
            {
                M(Viseme.sil),
                M(Viseme.PP,  T("*MTH_U*", 0.3f)),
                M(Viseme.FF,  T("*MTH_I*", 0.3f)),
                M(Viseme.TH,  T("*MTH_I*", 0.2f), T("*MTH_A*", 0.1f)),
                M(Viseme.DD,  T("*MTH_A*", 0.3f)),
                M(Viseme.kk,  T("*MTH_A*", 0.2f)),
                M(Viseme.CH,  T("*MTH_I*", 0.4f)),
                M(Viseme.SS,  T("*MTH_I*", 0.4f)),
                M(Viseme.nn,  T("*MTH_A*", 0.2f)),
                M(Viseme.RR,  T("*MTH_O*", 0.3f)),
                M(Viseme.aa,  T("*MTH_A*", 1.0f)),
                M(Viseme.E,   T("*MTH_E*", 1.0f)),
                M(Viseme.ih,  T("*MTH_I*", 1.0f)),
                M(Viseme.oh,  T("*MTH_O*", 1.0f)),
                M(Viseme.ou,  T("*MTH_U*", 1.0f)),
            };
        }

        // ─────────────────────────────────────────────────────────────────
        //  Daz3D Genesis 8/9 preset (eCTRL FACS viseme morphs)
        // ─────────────────────────────────────────────────────────────────
        private static VisemeBlendShapeMapping[] CreateDaz3DPreset()
        {
            return new[]
            {
                M(Viseme.sil),
                M(Viseme.PP,  T("*vBMP*", 1.0f)),
                M(Viseme.FF,  T("*vFV*", 1.0f)),
                M(Viseme.TH,  T("*vTH*", 1.0f)),
                M(Viseme.DD,  T("*vSTN*", 0.7f)),
                M(Viseme.kk,  T("*vGK*", 1.0f)),
                M(Viseme.CH,  T("*vSH*", 1.0f)),
                M(Viseme.SS,  T("*vSH*", 0.8f)),
                M(Viseme.nn,  T("*vSTN*", 1.0f)),
                M(Viseme.RR,  T("*vR*", 1.0f)),
                M(Viseme.aa,  T("*vAA*", 1.0f)),
                M(Viseme.E,   T("*vEE*", 1.0f)),
                M(Viseme.ih,  T("*vIH*", 1.0f)),
                M(Viseme.oh,  T("*vOH*", 1.0f)),
                M(Viseme.ou,  T("*vOU*", 1.0f)),
            };
        }

        // ─────────────────────────────────────────────────────────────────
        //  ARKit preset (multiple weighted blend shapes per viseme)
        // ─────────────────────────────────────────────────────────────────
        private static VisemeBlendShapeMapping[] CreateARKitPreset()
        {
            return new[]
            {
                M(Viseme.sil),
                M(Viseme.PP,  T("*mouthClose*", 0.8f), T("*mouthPucker*", 0.3f)),
                M(Viseme.FF,  T("*mouthLowerDownLeft*", 0.3f), T("*mouthLowerDownRight*", 0.3f),
                              T("*mouthUpperUpLeft*", 0.2f), T("*mouthUpperUpRight*", 0.2f)),
                M(Viseme.TH,  T("*jawOpen*", 0.1f), T("*tongueOut*", 0.4f),
                              T("*mouthLowerDownLeft*", 0.2f), T("*mouthLowerDownRight*", 0.2f)),
                M(Viseme.DD,  T("*jawOpen*", 0.2f), T("*mouthClose*", 0.3f)),
                M(Viseme.kk,  T("*jawOpen*", 0.25f)),
                M(Viseme.CH,  T("*mouthFunnel*", 0.5f), T("*jawOpen*", 0.15f)),
                M(Viseme.SS,  T("*jawOpen*", 0.1f), T("*mouthSmileLeft*", 0.2f), T("*mouthSmileRight*", 0.2f)),
                M(Viseme.nn,  T("*jawOpen*", 0.1f), T("*mouthClose*", 0.4f)),
                M(Viseme.RR,  T("*jawOpen*", 0.2f), T("*mouthPucker*", 0.3f)),
                M(Viseme.aa,  T("*jawOpen*", 0.7f), T("*mouthLowerDownLeft*", 0.4f), T("*mouthLowerDownRight*", 0.4f)),
                M(Viseme.E,   T("*jawOpen*", 0.3f), T("*mouthSmileLeft*", 0.3f), T("*mouthSmileRight*", 0.3f)),
                M(Viseme.ih,  T("*jawOpen*", 0.15f), T("*mouthSmileLeft*", 0.5f), T("*mouthSmileRight*", 0.5f)),
                M(Viseme.oh,  T("*jawOpen*", 0.5f), T("*mouthPucker*", 0.5f)),
                M(Viseme.ou,  T("*jawOpen*", 0.2f), T("*mouthPucker*", 0.7f), T("*mouthFunnel*", 0.5f)),
            };
        }

        // ─────────────────────────────────────────────────────────────────
        //  FACS preset (matches *vXX* and *v_XX* style viseme morphs)
        // ─────────────────────────────────────────────────────────────────
        private static VisemeBlendShapeMapping[] CreateFacsPreset()
        {
            return new[]
            {
                M(Viseme.sil),
                M(Viseme.PP,  T("*vM", 1.0f),   T("*v_M", 1.0f)),
                M(Viseme.FF,  T("*vF", 1.0f),   T("*v_F", 1.0f)),
                M(Viseme.TH,  T("*vTH", 1.0f),  T("*v_TH", 1.0f)),
                M(Viseme.DD,  T("*vT", 1.0f),   T("*v_T", 1.0f)),
                M(Viseme.kk,  T("*vK", 1.0f),   T("*v_K", 1.0f)),
                M(Viseme.CH,  T("*vSH", 1.0f),  T("*v_SH", 1.0f)),
                M(Viseme.SS,  T("*vS", 1.0f),   T("*v_S", 1.0f)),
                M(Viseme.nn,  T("*vL", 1.0f),   T("*v_L", 1.0f)),
                M(Viseme.RR,  T("*vER", 1.0f),  T("*v_ER", 1.0f)),
                M(Viseme.aa,  T("*vAA", 1.0f),  T("*v_AA", 1.0f)),
                M(Viseme.E,   T("*vEE", 0.6f),  T("*v_EE", 0.6f),  T("*vEH", 0.4f),  T("*v_EH", 0.4f)),
                M(Viseme.ih,  T("*vIH", 0.5f),  T("*v_IH", 0.5f),  T("*vIY", 0.5f),  T("*v_IY", 0.5f)),
                M(Viseme.oh,  T("*vOW", 1.0f),  T("*v_OW", 1.0f)),
                M(Viseme.ou,  T("*vUW", 0.7f),  T("*v_UW", 0.7f),  T("*vW", 0.3f),   T("*v_W", 0.3f)),
            };
        }

        // ─────────────────────────────────────────────────────────────────
        //  VRChat preset (vrc.v_sil, vrc.v_PP, vrc.v_AA, etc.)
        // ─────────────────────────────────────────────────────────────────
        private static VisemeBlendShapeMapping[] CreateVRChatPreset()
        {
            return new[]
            {
                M(Viseme.sil, T("*v_sil", 1.0f)),
                M(Viseme.PP,  T("*v_PP", 1.0f)),
                M(Viseme.FF,  T("*v_FF", 1.0f)),
                M(Viseme.TH,  T("*v_TH", 1.0f)),
                M(Viseme.DD,  T("*v_DD", 1.0f)),
                M(Viseme.kk,  T("*v_KK", 1.0f)),
                M(Viseme.CH,  T("*v_CH", 1.0f)),
                M(Viseme.SS,  T("*v_SS", 1.0f)),
                M(Viseme.nn,  T("*v_NN", 1.0f)),
                M(Viseme.RR,  T("*v_RR", 1.0f)),
                M(Viseme.aa,  T("*v_AA", 1.0f)),
                M(Viseme.E,   T("*v_E", 1.0f)),
                M(Viseme.ih,  T("*v_IH", 1.0f)),
                M(Viseme.oh,  T("*v_OH", 1.0f)),
                M(Viseme.ou,  T("*v_OU", 1.0f)),
            };
        }

        // ── Helpers for concise preset definitions ──
        private static BlendShapeTarget T(string name, float weight) => new(name, weight);
        private static VisemeBlendShapeMapping M(Viseme v, params BlendShapeTarget[] t) => new(v, t);

        // ─────────────────────────────────────────────────────────────────
        //  Wildcard blend shape resolution
        // ─────────────────────────────────────────────────────────────────

        /// <summary>
        /// Resolve a blend shape index on a mesh, supporting '*' wildcards.
        /// Returns -1 if no match is found.
        /// </summary>
        public static int ResolveBlendShapeIndex(Mesh mesh, string pattern)
        {
            if (mesh == null || string.IsNullOrEmpty(pattern))
                return -1;

            // Fast path: no wildcards
            if (!pattern.Contains('*'))
                return mesh.GetBlendShapeIndex(pattern);

            // Wildcard matching
            int starIndex = pattern.IndexOf('*');
            string prefix = pattern.Substring(0, starIndex);
            string rest = pattern.Substring(starIndex + 1);

            bool hasSecondStar = rest.Contains('*');
            string middle = null;
            string suffix = null;

            if (hasSecondStar)
            {
                // *middle* pattern
                int secondStar = rest.IndexOf('*');
                middle = rest.Substring(0, secondStar);
            }
            else
            {
                suffix = rest;
            }

            int count = mesh.blendShapeCount;
            for (int i = 0; i < count; i++)
            {
                string name = mesh.GetBlendShapeName(i);

                if (prefix.Length > 0 && !name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    continue;

                if (suffix != null && suffix.Length > 0 && !name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                    continue;

                if (middle != null && middle.Length > 0 && !name.Contains(middle, StringComparison.OrdinalIgnoreCase))
                    continue;

                // prefix* or *suffix or prefix*suffix or *middle* — all matched
                if (prefix.Length == 0 && suffix is { Length: 0 } && middle == null)
                    continue; // bare '*' matches nothing useful

                return i;
            }

            return -1;
        }
    }
}

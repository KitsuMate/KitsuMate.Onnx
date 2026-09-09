using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync
{
    public sealed class BlendShapeVisemeTarget : VisemeTarget
    {
        [SerializeField] private GameObject targetRoot;
        [SerializeField] private bool includeInactive;
        [SerializeField] private VisemeBlendShapeProfile profile;
        [SerializeField, Range(0f, 1f)] private float maxWeight = 1f;
        [SerializeField, Range(0.01f, 0.5f)] private float smoothTime = 0.05f;

        private sealed class Binding
        {
            public SkinnedMeshRenderer Renderer;
            public int Index;
            public readonly float[] Weights = new float[15];
        }

        private readonly List<Binding> bindings = new();
        private readonly float[] current = new float[15];
        private readonly float[] desired = new float[15];
        private readonly float[] velocities = new float[15];

        public void Configure(GameObject root, VisemeBlendShapeProfile mapping, float smoothing = 0.05f, float weight = 1f)
        {
            targetRoot = root;
            profile = mapping;
            smoothTime = Mathf.Clamp(smoothing, 0.01f, 0.5f);
            maxWeight = Mathf.Clamp01(weight);
            Refresh();
        }

        private void OnEnable() => Refresh();
        private void OnDisable() => ResetVisemes();

        public void Refresh()
        {
            ResetVisemes();
            bindings.Clear();
            if (profile == null) return;
            var root = targetRoot != null ? targetRoot : gameObject;
            foreach (var renderer in root.GetComponentsInChildren<SkinnedMeshRenderer>(includeInactive))
            {
                if (renderer.sharedMesh == null) continue;
                foreach (Viseme viseme in Enum.GetValues(typeof(Viseme)))
                foreach (var target in profile.GetTargetsForViseme(viseme))
                {
                    int index = VisemeBlendShapeProfile.ResolveBlendShapeIndex(renderer.sharedMesh, target.BlendShapeName);
                    if (index < 0) continue;
                    var binding = bindings.Find(b => b.Renderer == renderer && b.Index == index);
                    if (binding == null)
                    {
                        binding = new Binding { Renderer = renderer, Index = index };
                        bindings.Add(binding);
                    }
                    binding.Weights[(int)viseme] = target.Weight;
                }
            }
        }

        public override void ApplyViseme(VisemeFrame frame)
        {
            Array.Clear(desired, 0, desired.Length);
            if (frame.Viseme != Viseme.sil && (uint)frame.Viseme < desired.Length)
                desired[(int)frame.Viseme] = Mathf.Clamp01(frame.Weight) * maxWeight;
        }

        public override void ResetVisemes()
        {
            Array.Clear(current, 0, current.Length);
            Array.Clear(desired, 0, desired.Length);
            Array.Clear(velocities, 0, velocities.Length);
            foreach (var binding in bindings)
                if (binding.Renderer != null) binding.Renderer.SetBlendShapeWeight(binding.Index, 0f);
        }

        private void LateUpdate()
        {
            for (int i = 0; i < current.Length; i++)
                current[i] = Mathf.SmoothDamp(current[i], desired[i], ref velocities[i], smoothTime);
            foreach (var binding in bindings)
            {
                if (binding.Renderer == null) continue;
                float weight = 0f;
                for (int i = 0; i < current.Length; i++) weight += current[i] * binding.Weights[i];
                binding.Renderer.SetBlendShapeWeight(binding.Index, Mathf.Clamp01(weight) * 100f);
            }
        }
    }
}

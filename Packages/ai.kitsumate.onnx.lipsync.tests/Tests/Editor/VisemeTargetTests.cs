using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace KitsuMate.Onnx.LipSync.Tests
{
    public sealed class VisemeTargetTests
    {
        private static IEnumerator WaitSeconds(float seconds)
        {
            float end = Time.time + seconds;
            while (Time.time < end) yield return null;
        }

        [UnityTest]
        public IEnumerator SharedBlendShapeReceivesCombinedWeightAndResetsOnDisable()
        {
            yield return new EnterPlayMode();
            var root = new GameObject("Viseme target test");
            var mesh = new Mesh();
            var profile = ScriptableObject.CreateInstance<VisemeBlendShapeProfile>();
            try
            {
                mesh.vertices = new[] { Vector3.zero, Vector3.right, Vector3.up };
                mesh.triangles = new[] { 0, 1, 2 };
                mesh.AddBlendShapeFrame("mouth", 100, new[] { Vector3.up, Vector3.up, Vector3.up }, new Vector3[3], new Vector3[3]);
                var renderer = root.AddComponent<SkinnedMeshRenderer>();
                renderer.sharedMesh = mesh;
                profile.Mappings = new[]
                {
                    new VisemeBlendShapeMapping(Viseme.aa, new BlendShapeTarget("mouth")),
                    new VisemeBlendShapeMapping(Viseme.oh, new BlendShapeTarget("mouth"))
                };
                var target = root.AddComponent<BlendShapeVisemeTarget>();
                target.Configure(root, profile, .01f);
                target.ApplyViseme(new VisemeFrame(Viseme.aa, 0, 1, .7f));
                yield return WaitSeconds(.2f);
                Assert.That(renderer.GetBlendShapeWeight(0), Is.EqualTo(70).Within(1));
                target.ApplyViseme(new VisemeFrame(Viseme.oh, 0, 1, .4f));
                yield return WaitSeconds(.2f);
                Assert.That(renderer.GetBlendShapeWeight(0), Is.EqualTo(40).Within(1));
                target.enabled = false;
                Assert.That(renderer.GetBlendShapeWeight(0), Is.Zero);
            }
            finally
            {
                Object.Destroy(root);
                Object.Destroy(mesh);
                Object.Destroy(profile);
            }
            yield return new ExitPlayMode();
        }
    }
}

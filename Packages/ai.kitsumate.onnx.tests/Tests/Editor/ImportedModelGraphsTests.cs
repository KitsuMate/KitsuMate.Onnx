using System;
using System.IO;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class ImportedModelGraphsTests
    {
        [Test]
        public void RenamedModelSetKeepsInstallationOwnershipButDuplicateCannotReplaceIt()
        {
            string assetFolder = "Assets/KitsuMateOnnxOwnershipTest-" + Guid.NewGuid().ToString("N");
            string originalPath = assetFolder + "/Original.asset";
            string renamedPath = assetFolder + "/Renamed.asset";
            string duplicatePath = assetFolder + "/Duplicate.asset";
            ImportedModelGraphsTestModelSet original = null;
            try
            {
                Directory.CreateDirectory(assetFolder);
                original = ScriptableObject.CreateInstance<ImportedModelGraphsTestModelSet>();
                AssetDatabase.CreateAsset(original, originalPath);
                string owner = AssetDatabase.AssetPathToGUID(originalPath);
                original.Download.installationOwnerGuid = owner;
                original.Download.installationFolder = "model-sets/" + owner;
                EditorUtility.SetDirty(original);
                AssetDatabase.SaveAssetIfDirty(original);

                Assert.That(AssetDatabase.MoveAsset(originalPath, renamedPath), Is.Empty);
                Assert.That(AssetDatabase.AssetPathToGUID(renamedPath), Is.EqualTo(owner));
                Assert.DoesNotThrow(() => original.Installation(Path.GetTempPath()));

                Assert.That(AssetDatabase.CopyAsset(renamedPath, duplicatePath), Is.True);
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                var duplicate = AssetDatabase.LoadAssetAtPath<ImportedModelGraphsTestModelSet>(duplicatePath);
                Assert.That(AssetDatabase.AssetPathToGUID(duplicatePath), Is.Not.EqualTo(owner));
                Assert.Throws<InvalidOperationException>(() => duplicate.Installation(Path.GetTempPath()));
            }
            finally
            {
                AssetDatabase.DeleteAsset(assetFolder);
                if (original != null && !AssetDatabase.Contains(original)) UnityEngine.Object.DestroyImmediate(original);
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            }
        }

        [Test]
        public void ModelAndCompanionsRoundTripBetweenFilesAndAssets()
        {
            const string graphBase64 = "CAoSDG1vdmUtZml4dHVyZTqaAQocCgVpbnB1dAoGd2VpZ2h0EgZvdXRwdXQiA0FkZBIFZmlyc3QqSAgBEAFCBndlaWdodGoeCghsb2NhdGlvbhISd2VpZ2h0cy9zaGFyZWQuYmluagsKBm9mZnNldBIBMGoLCgZsZW5ndGgSATRwAVoTCgVpbnB1dBIKCggIARIECgIIAWIUCgZvdXRwdXQSCgoICAESBAoCCAFCAhAV";
            string id = Guid.NewGuid().ToString("N");
            string assetFolder = "Assets/KitsuMateOnnxMoveTest-" + id;
            string sourceRoot = Path.Combine(Path.GetTempPath(), "kitsumate-move-source-" + id);
            string dataRoot = Path.Combine(Path.GetTempPath(), "kitsumate-move-data-" + id);
            var set = ScriptableObject.CreateInstance<ImportedModelGraphsTestModelSet>();
            try
            {
                string graph = Path.Combine(sourceRoot, "model.onnx");
                string weights = Path.Combine(sourceRoot, "weights", "shared.bin");
                string tokenizer = Path.Combine(sourceRoot, "tokenizer.json");
                Directory.CreateDirectory(Path.GetDirectoryName(weights));
                File.WriteAllBytes(graph, Convert.FromBase64String(graphBase64));
                File.WriteAllBytes(weights, new byte[] { 1, 2, 3, 4 });
                File.WriteAllText(tokenizer, "{}");
                set.Model.ConfigureFile(graph, "", null, null);
                set.Tokenizer.ConfigureFile(tokenizer);
                Directory.CreateDirectory(assetFolder);
                AssetDatabase.CreateAsset(set, assetFolder + "/ModelSet.asset");
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);

                ImportedModelGraphs.MoveAssignedFiles(set);

                Assert.That(set.Model.Kind, Is.EqualTo(OnnxModelReference.SourceKind.Asset));
                Assert.That(set.Model.Asset, Is.Not.Null);
                Assert.That(set.Tokenizer.Kind, Is.EqualTo(OnnxModelReference.SourceKind.Asset));
                string importedGraph = AssetDatabase.GetAssetPath(set.Model.Asset);
                Assert.That(importedGraph, Is.EqualTo(assetFolder + "/ModelSet Files/model-0/model.onnx"));
                Assert.That(File.Exists(Path.Combine(Path.GetDirectoryName(importedGraph), "weights", "shared.bin")), Is.True);
                Assert.That(AssetDatabase.GetAssetPath(set.Tokenizer.Asset),
                    Is.EqualTo(assetFolder + "/ModelSet Files/text-0/tokenizer.json.bytes"));
                Assert.That(File.Exists(graph), Is.False);
                Assert.That(File.Exists(tokenizer), Is.False);

                ImportedModelGraphs.MoveAssignedFilesToData(set, dataRoot);

                Assert.That(set.Model.Kind, Is.EqualTo(OnnxModelReference.SourceKind.File));
                Assert.That(set.Tokenizer.Kind, Is.EqualTo(OnnxModelReference.SourceKind.File));
                string movedGraph = set.Model.ResolveModelPath();
                Assert.That(movedGraph, Does.StartWith(Path.GetFullPath(dataRoot)));
                Assert.That(File.Exists(movedGraph), Is.True);
                Assert.That(File.Exists(Path.Combine(Path.GetDirectoryName(movedGraph), "weights", "shared.bin")), Is.True);
                Assert.That(File.Exists(set.Tokenizer.ResolveFile()), Is.True);
                Assert.That(AssetDatabase.LoadAssetAtPath<OnnxModelAsset>(importedGraph), Is.Null);
            }
            finally
            {
                AssetDatabase.DeleteAsset(assetFolder);
                if (Directory.Exists(sourceRoot)) Directory.Delete(sourceRoot, true);
                if (Directory.Exists(dataRoot)) Directory.Delete(dataRoot, true);
                if (set != null && !AssetDatabase.Contains(set)) UnityEngine.Object.DestroyImmediate(set);
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            }
        }
    }
}

using System;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Editor
{
    [CustomEditor(typeof(LipSyncPlayer))]
    internal class LipSyncPlayerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var player = (LipSyncPlayer)target;

            if (!Application.isPlaying)
                return;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Preview", EditorStyles.boldLabel);

            bool hasClip = player.AudioSource != null && player.AudioSource.clip != null;
            bool hasProfile = player.Profile != null;
            bool hasEngine = player.Engine != null;
            bool canPreview = hasProfile && hasEngine && (player.Mode == LipSyncPlayer.PlaybackMode.Realtime || hasClip);

            EditorGUI.BeginDisabledGroup(!canPreview);

            DrawModeStatus(player, hasClip);

            if (player.IsAnalyzing)
            {
                EditorGUILayout.HelpBox(
                    player.Mode == LipSyncPlayer.PlaybackMode.Realtime
                        ? "Analyzing recent audio window..."
                        : "Analyzing audio...",
                    MessageType.Info);
            }
            else if (player.AudioSource != null && player.AudioSource.isPlaying)
            {
                EditorGUILayout.BeginHorizontal();
                string label = player.Mode == LipSyncPlayer.PlaybackMode.Realtime
                    ? $"Realtime — Viseme: {player.CurrentViseme}"
                    : $"Timeline — Viseme: {player.CurrentViseme}";
                EditorGUILayout.LabelField(label);
                if (GUILayout.Button("Stop", GUILayout.Width(60)))
                {
                    player.Stop();
                }
                EditorGUILayout.EndHorizontal();
                Repaint();
            }
            else
            {
                string playLabel = player.Mode == LipSyncPlayer.PlaybackMode.Realtime
                    ? "Play Realtime Lip Sync"
                    : player.PlayDelay > 0f
                        ? $"Play Lip Sync ({player.PlayDelay:F2}s delay)"
                        : "Play Lip Sync";

                if (GUILayout.Button(playLabel))
                {
                    PlayPreview(player);
                }
            }

            EditorGUI.EndDisabledGroup();

            if (!hasClip)
            {
                if (player.Mode == LipSyncPlayer.PlaybackMode.Timeline)
                    EditorGUILayout.HelpBox("Assign an AudioSource with a clip to preview timeline mode.", MessageType.Warning);
                else
                    EditorGUILayout.HelpBox("Assign an AudioSource to preview realtime mode. A microphone clip routed through that source also works.", MessageType.Warning);
            }
            else if (!hasProfile)
                EditorGUILayout.HelpBox("Assign a VisemeBlendShapeProfile to preview.", MessageType.Warning);
            else if (!hasEngine)
                EditorGUILayout.HelpBox("Assign a LipSyncEngine to preview.", MessageType.Warning);
        }

        private static void DrawModeStatus(LipSyncPlayer player, bool hasClip)
        {
            switch (player.Mode)
            {
                case LipSyncPlayer.PlaybackMode.Timeline:
                    if (player.PlayDelay > 0f)
                    {
                        EditorGUILayout.HelpBox(
                            $"Timeline mode will analyze first and then start playback with the configured {player.PlayDelay:F2}s delay.",
                            MessageType.None);
                    }
                    else if (hasClip)
                    {
                        EditorGUILayout.HelpBox(
                            "Timeline mode follows the current audio clock. If analysis completes late, playback jumps straight to the current clip time.",
                            MessageType.None);
                    }
                    break;
                case LipSyncPlayer.PlaybackMode.Realtime:
                    EditorGUILayout.HelpBox(
                        "Realtime mode reads recent samples from the assigned AudioSource and applies the newest detected viseme while audio is playing.",
                        MessageType.None);
                    break;
            }
        }

        public override bool RequiresConstantRepaint()
        {
            if (!Application.isPlaying)
                return false;

            var player = (LipSyncPlayer)target;
            return player.AudioSource != null && player.AudioSource.isPlaying;
        }

        private static async void PlayPreview(LipSyncPlayer player)
        {
            try
            {
                if (player.Mode == LipSyncPlayer.PlaybackMode.Timeline && player.PlayDelay > 0f)
                {
                    await player.PlayDelayedAsync(player.AudioSource.clip);
                }
                else
                {
                    await player.PlayAsync(player.AudioSource.clip);
                }
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }
    }
}

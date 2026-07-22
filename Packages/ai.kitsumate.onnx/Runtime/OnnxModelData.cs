using System;
using UnityEngine;

namespace KitsuMate.Onnx
{
    /// <summary>
    /// Container for ONNX model binary data.
    /// Stored as a hidden sub-asset to avoid inspector serialization overhead.
    /// </summary>
    [PreferBinarySerialization]
    public class OnnxModelData : ScriptableObject
    {
        [SerializeField, HideInInspector] 
        private byte[] _data = Array.Empty<byte>();

        /// <summary>
        /// Gets the raw model bytes as ReadOnlyMemory.
        /// </summary>
        public ReadOnlyMemory<byte> Data => new ReadOnlyMemory<byte>(_data ?? Array.Empty<byte>());

        /// <summary>
        /// Gets a copy of the model bytes.
        /// </summary>
        public byte[] GetDataCopy()
        {
            return _data != null ? (byte[])_data.Clone() : Array.Empty<byte>();
        }

        /// <summary>
        /// Gets the raw byte array reference. Use with caution - do not modify.
        /// </summary>
        internal byte[] GetDataRef() => _data ?? Array.Empty<byte>();

        /// <summary>
        /// Whether this container has any data.
        /// </summary>
        public bool HasData => _data != null && _data.Length > 0;

        /// <summary>
        /// Size in bytes.
        /// </summary>
        public long Size => _data?.Length ?? 0;

#if UNITY_EDITOR
        /// <summary>
        /// Sets the model data. Editor-only.
        /// </summary>
        public void SetData(byte[] data)
        {
            _data = data ?? Array.Empty<byte>();
        }
#endif
    }
}

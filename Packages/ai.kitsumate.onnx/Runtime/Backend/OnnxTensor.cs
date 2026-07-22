using System;

namespace KitsuMate.Onnx
{
    /// <summary>
    /// Backend-agnostic tensor representation.
    /// Wraps tensor data for passing between layers without exposing backend-specific types.
    /// </summary>
    public class OnnxTensor : IDisposable
    {
        /// <summary>Name of this tensor (for named inputs/outputs).</summary>
        public string Name { get; set; }
        
        /// <summary>Shape of the tensor (e.g., [1, 3, 224, 224]).</summary>
        public int[] Shape { get; }
        
        /// <summary>Element type of the tensor data.</summary>
        public OnnxTensorElementType ElementType { get; }
        
        /// <summary>Raw data buffer. Type depends on ElementType.</summary>
        public Array Data { get; }
        
        /// <summary>Total number of elements in the tensor.</summary>
        public int Length { get; }
        
        private bool _disposed;
        
        public OnnxTensor(int[] shape, OnnxTensorElementType elementType, Array data, string name = null)
        {
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            ElementType = elementType;
            Data = data ?? throw new ArgumentNullException(nameof(data));
            Name = name;
            
            int length = 1;
            foreach (var dim in shape)
                length *= dim;
            Length = length;
        }
        
        /// <summary>Get data as float array. Throws if ElementType is not Float.</summary>
        public float[] AsFloatArray()
        {
            if (ElementType != OnnxTensorElementType.Float)
                throw new InvalidOperationException($"Tensor is {ElementType}, not Float");
            return (float[])Data;
        }
        
        /// <summary>Get data as int array. Throws if ElementType is not Int32.</summary>
        public int[] AsIntArray()
        {
            if (ElementType != OnnxTensorElementType.Int32)
                throw new InvalidOperationException($"Tensor is {ElementType}, not Int32");
            return (int[])Data;
        }
        
        /// <summary>Get data as long array. Throws if ElementType is not Int64.</summary>
        public long[] AsLongArray()
        {
            if (ElementType != OnnxTensorElementType.Int64)
                throw new InvalidOperationException($"Tensor is {ElementType}, not Int64");
            return (long[])Data;
        }
        
        /// <summary>Get data as byte array. Throws if ElementType is not UInt8.</summary>
        public byte[] AsByteArray()
        {
            if (ElementType != OnnxTensorElementType.UInt8)
                throw new InvalidOperationException($"Tensor is {ElementType}, not UInt8");
            return (byte[])Data;
        }

        /// <summary>Get data as Boolean array. Throws if ElementType is not Bool.</summary>
        public bool[] AsBoolArray()
        {
            if (ElementType != OnnxTensorElementType.Bool)
                throw new InvalidOperationException($"Tensor is {ElementType}, not Bool");
            return (bool[])Data;
        }
        
        /// <summary>Get data as ushort (FP16) array. Throws if ElementType is not Float16.</summary>
        /// <remarks>Uses ushort as Half is not available in netstandard2.0. Use BitConverter for conversion.</remarks>
        public ushort[] AsFloat16Array()
        {
            if (ElementType != OnnxTensorElementType.Float16)
                throw new InvalidOperationException($"Tensor is {ElementType}, not Float16");
            return (ushort[])Data;
        }
        
        // Factory methods
        public static OnnxTensor FromArray(float[] data, int[] shape, string name = null)
            => new(shape, OnnxTensorElementType.Float, data, name);
        
        public static OnnxTensor FromArray(int[] data, int[] shape, string name = null)
            => new(shape, OnnxTensorElementType.Int32, data, name);
        
        public static OnnxTensor FromArray(long[] data, int[] shape, string name = null)
            => new(shape, OnnxTensorElementType.Int64, data, name);
        
        public static OnnxTensor FromArray(byte[] data, int[] shape, string name = null)
            => new(shape, OnnxTensorElementType.UInt8, data, name);

        public static OnnxTensor FromArray(bool[] data, int[] shape, string name = null)
            => new(shape, OnnxTensorElementType.Bool, data, name);
        
        public static OnnxTensor FromArray(ushort[] data, int[] shape, string name = null)
            => new(shape, OnnxTensorElementType.Float16, data, name);
        
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // Data array will be GC'd
        }
    }
    
    /// <summary>
    /// Named value pair for session inputs.
    /// </summary>
    public readonly struct OnnxNamedValue
    {
        public string Name { get; }
        public OnnxTensor Value { get; }
        
        public OnnxNamedValue(string name, OnnxTensor value)
        {
            Name = name;
            Value = value;
        }
    }
    
    /// <summary>
    /// Tensor element types supported across backends.
    /// </summary>
    public enum OnnxTensorElementType
    {
        Undefined = 0,
        Float = 1,
        UInt8 = 2,
        Int8 = 3,
        UInt16 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        String = 8,
        Bool = 9,
        Float16 = 10,
        Double = 11,
        UInt32 = 12,
        UInt64 = 13,
        BFloat16 = 16
    }
}

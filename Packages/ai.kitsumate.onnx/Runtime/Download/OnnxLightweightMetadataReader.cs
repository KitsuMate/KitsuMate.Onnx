using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace KitsuMate.Onnx
{
    /// <summary>Streams ONNX protobuf metadata and skips initializer payloads. Never creates an ORT session.</summary>
    // Uses a local wire reader because Google.Protobuf's nested-limit APIs are not public in the bundled assembly.
    public static class OnnxLightweightMetadataReader
    {
        public sealed class Result
        {
            public string GraphName = string.Empty;
            public string Domain = string.Empty;
            public string Description = string.Empty;
            public string Producer = string.Empty;
            public long OpsetVersion;
            public readonly List<OnnxModelAsset.TensorInfo> Inputs = new();
            public readonly List<OnnxModelAsset.TensorInfo> Outputs = new();
            public readonly List<OnnxModelAsset.MetadataEntry> CustomMetadata = new();
            internal bool GraphFound;
        }

        public static Result Read(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Model path is required.", nameof(path));
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan);
            using var input = new ProtoReader(stream);
            var result = new Result();
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (Field(tag))
                {
                    case 2: result.Producer = input.ReadString(); break;
                    case 4: result.Domain = input.ReadString(); break;
                    case 6: result.Description = input.ReadString(); break;
                    case 7: ReadMessage(input, nested => ReadGraph(nested, result)); break;
                    case 8: ReadMessage(input, nested => ReadOpset(nested, result)); break;
                    case 14: ReadMessage(input, nested => ReadMetadataEntry(nested, result)); break;
                    default: input.SkipLastField(); break;
                }
            }
            if (!result.GraphFound || result.Outputs.Count == 0)
                throw new InvalidDataException("The file is not a supported ONNX protobuf model or contains no graph outputs.");
            return result;
        }

        private static void ReadGraph(ProtoReader input, Result result)
        {
            result.GraphFound = true;
            var initializerNames = new HashSet<string>(StringComparer.Ordinal);
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (Field(tag))
                {
                    case 2: result.GraphName = input.ReadString(); break;
                    case 5: ReadMessage(input, nested => ReadInitializerName(nested, initializerNames)); break;
                    case 10: if (string.IsNullOrWhiteSpace(result.Description)) result.Description = input.ReadString(); else input.ReadString(); break;
                    case 11: ReadMessage(input, nested => result.Inputs.Add(ReadValueInfo(nested))); break;
                    case 12: ReadMessage(input, nested => result.Outputs.Add(ReadValueInfo(nested))); break;
                    default: input.SkipLastField(); break; // nodes and initializer byte payloads
                }
            }
            result.Inputs.RemoveAll(inputInfo => initializerNames.Contains(inputInfo.Name));
        }

        private static void ReadInitializerName(ProtoReader input, HashSet<string> names)
        {
            string name = string.Empty; uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                if (Field(tag) == 8) name = input.ReadString();
                else input.SkipLastField();
            }
            if (!string.IsNullOrEmpty(name)) names.Add(name);
        }

        private static OnnxModelAsset.TensorInfo ReadValueInfo(ProtoReader input)
        {
            string name = string.Empty, elementType = string.Empty; int[] shape = Array.Empty<int>(); uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (Field(tag))
                {
                    case 1: name = input.ReadString(); break;
                    case 2: ReadMessage(input, nested => ReadType(nested, out elementType, out shape)); break;
                    default: input.SkipLastField(); break;
                }
            }
            string description = shape.Length == 0 ? "?" : string.Join("x", Array.ConvertAll(shape, x => x < 0 ? "*" : x.ToString()));
            return new OnnxModelAsset.TensorInfo(name, elementType, shape, description);
        }

        private static void ReadType(ProtoReader input, out string elementType, out int[] shape)
        {
            elementType = string.Empty; shape = Array.Empty<int>(); uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                if (Field(tag) == 1)
                {
                    string localType = string.Empty; int[] localShape = Array.Empty<int>();
                    ReadMessage(input, nested => ReadTensorType(nested, out localType, out localShape));
                    elementType = localType; shape = localShape;
                }
                else input.SkipLastField();
            }
        }

        private static void ReadTensorType(ProtoReader input, out string elementType, out int[] shape)
        {
            int type = 0; var dimensions = new List<int>(); uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (Field(tag))
                {
                    case 1: type = input.ReadEnum(); break;
                    case 2: ReadMessage(input, nested => ReadShape(nested, dimensions)); break;
                    default: input.SkipLastField(); break;
                }
            }
            elementType = TensorTypeName(type); shape = dimensions.ToArray();
        }

        private static void ReadShape(ProtoReader input, List<int> dimensions)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
                if (Field(tag) == 1) ReadMessage(input, nested => dimensions.Add(ReadDimension(nested)));
                else input.SkipLastField();
        }

        private static int ReadDimension(ProtoReader input)
        {
            int value = -1; uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (Field(tag))
                {
                    case 1: long raw = input.ReadInt64(); value = raw > int.MaxValue ? int.MaxValue : raw < int.MinValue ? int.MinValue : (int)raw; break;
                    case 2: input.ReadString(); value = -1; break;
                    default: input.SkipLastField(); break;
                }
            }
            return value;
        }

        private static void ReadOpset(ProtoReader input, Result result)
        {
            string domain = string.Empty; long version = 0; uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                if (Field(tag) == 1) domain = input.ReadString();
                else if (Field(tag) == 2) version = input.ReadInt64();
                else input.SkipLastField();
            }
            if (string.IsNullOrEmpty(domain)) result.OpsetVersion = version;
        }

        private static void ReadMetadataEntry(ProtoReader input, Result result)
        {
            string key = string.Empty, value = string.Empty; uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                if (Field(tag) == 1) key = input.ReadString();
                else if (Field(tag) == 2) value = input.ReadString();
                else input.SkipLastField();
            }
            if (!string.IsNullOrEmpty(key)) result.CustomMetadata.Add(new OnnxModelAsset.MetadataEntry(key, value));
        }

        private static void ReadMessage(ProtoReader input, Action<ProtoReader> reader)
        {
            long oldLimit = input.PushLimit(input.ReadLength());
            try { reader(input); }
            finally { input.PopLimit(oldLimit); }
        }

        private static int Field(uint tag) => (int)(tag >> 3);

        private sealed class ProtoReader : IDisposable
        {
            private readonly Stream stream;
            private long limit;
            private uint lastTag;
            public ProtoReader(Stream stream) { this.stream = stream; limit = stream.Length; }
            public uint ReadTag() { if (stream.Position >= limit) return lastTag = 0; return lastTag = checked((uint)ReadVarint()); }
            public int ReadLength() { ulong value = ReadVarint(); if (value > int.MaxValue) throw new InvalidDataException("Protobuf field is too large."); return (int)value; }
            public long ReadInt64() => unchecked((long)ReadVarint());
            public int ReadEnum() => checked((int)ReadVarint());
            public string ReadString()
            {
                int length = ReadLength(); EnsureAvailable(length); var bytes = new byte[length]; int offset = 0;
                while (offset < length) { int read = stream.Read(bytes, offset, length - offset); if (read <= 0) throw new EndOfStreamException(); offset += read; }
                return Encoding.UTF8.GetString(bytes);
            }
            public long PushLimit(int byteLength)
            {
                long nested = checked(stream.Position + byteLength); if (nested > limit) throw new InvalidDataException("Nested protobuf message exceeds its parent field.");
                long previous = limit; limit = nested; return previous;
            }
            public void PopLimit(long previous) { if (stream.Position < limit) stream.Seek(limit - stream.Position, SeekOrigin.Current); limit = previous; }
            public void SkipLastField()
            {
                switch (lastTag & 7)
                {
                    case 0: ReadVarint(); break;
                    case 1: Skip(8); break;
                    case 2: Skip(ReadLength()); break;
                    case 5: Skip(4); break;
                    default: throw new InvalidDataException($"Unsupported protobuf wire type {lastTag & 7}.");
                }
            }
            private ulong ReadVarint()
            {
                ulong value = 0; for (int shift = 0; shift < 64; shift += 7)
                { int next = stream.ReadByte(); if (next < 0 || stream.Position > limit) throw new EndOfStreamException(); value |= (ulong)(next & 0x7f) << shift; if ((next & 0x80) == 0) return value; }
                throw new InvalidDataException("Invalid protobuf varint.");
            }
            private void Skip(long count) { EnsureAvailable(count); stream.Seek(count, SeekOrigin.Current); }
            private void EnsureAvailable(long count) { if (count < 0 || stream.Position + count > limit) throw new InvalidDataException("Protobuf field exceeds its containing message."); }
            public void Dispose() => stream.Dispose();
        }

        private static string TensorTypeName(int type) => type switch
        {
            1 => "System.Single", 2 => "System.Byte", 3 => "System.SByte", 4 => "System.UInt16",
            5 => "System.Int16", 6 => "System.Int32", 7 => "System.Int64", 8 => "System.String",
            9 => "System.Boolean", 10 => "System.Half", 11 => "System.Double",
            12 => "System.UInt32", 13 => "System.UInt64", 16 => "BFloat16",
            _ => $"ONNX.TensorProto.DataType({type})"
        };
    }
}

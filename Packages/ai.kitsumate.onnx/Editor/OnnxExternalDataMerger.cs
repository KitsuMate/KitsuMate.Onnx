#if UNITY_EDITOR
#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Merges external data (.onnx_data) into ONNX models at import time for cross-platform compatibility.
    /// </summary>
    public static class OnnxExternalDataMerger
    {
        private const int WireTypeVarint = 0;
        private const int WireTypeLengthDelimited = 2;
        
        private const int ModelProto_Graph = 7;
        private const int GraphProto_Initializer = 5;
        private const int TensorProto_RawData = 9;
        private const int TensorProto_ExternalData = 13;
        private const int TensorProto_DataLocation = 14;
        private const int StringStringEntryProto_Key = 1;
        private const int StringStringEntryProto_Value = 2;
        
        private const int DataLocation_Default = 0;
        private const int DataLocation_External = 1;

        public static byte[] MergeExternalData(byte[] modelBytes, string modelDirectory)
        {
            if (modelBytes == null || modelBytes.Length == 0)
                return modelBytes ?? Array.Empty<byte>();

            try
            {
                using var inputStream = new MemoryStream(modelBytes);
                using var outputStream = new MemoryStream();
                
                bool modified = ProcessMessage(inputStream, outputStream, modelDirectory, MessageType.Model);
                
                return modified ? outputStream.ToArray() : modelBytes;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OnnxExternalDataMerger] Failed to merge external data: {ex.Message}");
                return modelBytes;
            }
        }

        private enum MessageType
        {
            Model,
            Graph,
            Tensor,
            StringStringEntry,
            Other
        }

        private static bool ProcessMessage(MemoryStream input, MemoryStream output, string modelDir, MessageType messageType)
        {
            bool modified = false;
            if (messageType == MessageType.Tensor)
            {
                return ProcessTensorMessage(input, output, modelDir);
            }
            
            while (input.Position < input.Length)
            {
                int tag = ReadVarint32(input);
                if (tag == 0) break;
                
                int fieldNumber = tag >> 3;
                int wireType = tag & 0x7;
                
                WriteVarint32(output, tag);
                
                if (wireType == WireTypeVarint)
                {
                    long value = ReadVarint64(input);
                    WriteVarint64(output, value);
                }
                else if (wireType == WireTypeLengthDelimited)
                {
                    int length = ReadVarint32(input);
                    byte[] data = new byte[length];
                    input.Read(data, 0, length);
                    
                    MessageType nestedType = GetNestedMessageType(messageType, fieldNumber);
                    
                    if (nestedType != MessageType.Other)
                    {
                        using var nestedInput = new MemoryStream(data);
                        using var nestedOutput = new MemoryStream();
                        
                        bool nestedModified = ProcessMessage(nestedInput, nestedOutput, modelDir, nestedType);
                        
                        if (nestedModified)
                        {
                            modified = true;
                            byte[] modifiedData = nestedOutput.ToArray();
                            WriteVarint32(output, modifiedData.Length);
                            output.Write(modifiedData, 0, modifiedData.Length);
                        }
                        else
                        {
                            WriteVarint32(output, length);
                            output.Write(data, 0, length);
                        }
                    }
                    else
                    {
                        WriteVarint32(output, length);
                        output.Write(data, 0, length);
                    }
                }
                else
                {
                    SkipField(input, wireType);
                }
            }
            
            return modified;
        }

        private static MessageType GetNestedMessageType(MessageType parentType, int fieldNumber)
        {
            return parentType switch
            {
                MessageType.Model when fieldNumber == ModelProto_Graph => MessageType.Graph,
                MessageType.Graph when fieldNumber == GraphProto_Initializer => MessageType.Tensor,
                _ => MessageType.Other
            };
        }

        private static bool ProcessTensorMessage(MemoryStream input, MemoryStream output, string modelDir)
        {
            var fields = new List<(int tag, byte[] data)>();
            int dataLocation = DataLocation_Default;
            var externalDataEntries = new List<byte[]>();
            string? location = null;
            long offset = 0;
            long length = 0;
            
            long startPos = input.Position;
            
            while (input.Position < input.Length)
            {
                int tag = ReadVarint32(input);
                if (tag == 0) break;
                
                int fieldNumber = tag >> 3;
                int wireType = tag & 0x7;
                
                if (wireType == WireTypeVarint)
                {
                    long value = ReadVarint64(input);
                    
                    if (fieldNumber == TensorProto_DataLocation)
                    {
                        dataLocation = (int)value;
                    }
                    
                    using var fieldStream = new MemoryStream();
                    WriteVarint32(fieldStream, tag);
                    WriteVarint64(fieldStream, value);
                    fields.Add((tag, fieldStream.ToArray()));
                }
                else if (wireType == WireTypeLengthDelimited)
                {
                    int len = ReadVarint32(input);
                    byte[] data = new byte[len];
                    input.Read(data, 0, len);
                    
                    if (fieldNumber == TensorProto_ExternalData)
                    {
                        externalDataEntries.Add(data);
                        ParseExternalDataEntry(data, ref location, ref offset, ref length);
                    }
                    
                    using var fieldStream = new MemoryStream();
                    WriteVarint32(fieldStream, tag);
                    WriteVarint32(fieldStream, len);
                    fieldStream.Write(data, 0, len);
                    fields.Add((tag, fieldStream.ToArray()));
                }
                else
                {
                    SkipField(input, wireType);
                }
            }
            
            if (dataLocation != DataLocation_External || string.IsNullOrEmpty(location))
            {
                foreach (var field in fields)
                {
                    output.Write(field.data, 0, field.data.Length);
                }
                return false;
            }
            
            byte[]? externalBytes = LoadExternalData(modelDir, location!, offset, length);
            if (externalBytes == null)
            {
                Debug.LogWarning($"[OnnxExternalDataMerger] Failed to load external data from '{location}'");
                foreach (var field in fields)
                {
                    output.Write(field.data, 0, field.data.Length);
                }
                return false;
            }
            
            foreach (var field in fields)
            {
                int fieldTag = field.data[0];
                int fieldNumber = fieldTag >> 3;
                if (fieldNumber == TensorProto_ExternalData || fieldNumber == TensorProto_DataLocation)
                    continue;
                    
                output.Write(field.data, 0, field.data.Length);
            }
            
            int rawDataTag = (TensorProto_RawData << 3) | WireTypeLengthDelimited;
            WriteVarint32(output, rawDataTag);
            WriteVarint32(output, externalBytes.Length);
            output.Write(externalBytes, 0, externalBytes.Length);
            return true;
        }

        private static void ParseExternalDataEntry(byte[] data, ref string? location, ref long offset, ref long length)
        {
            using var stream = new MemoryStream(data);
            string? key = null;
            string? value = null;
            
            while (stream.Position < stream.Length)
            {
                int tag = ReadVarint32(stream);
                if (tag == 0) break;
                
                int fieldNumber = tag >> 3;
                int wireType = tag & 0x7;
                
                if (wireType == WireTypeLengthDelimited)
                {
                    int len = ReadVarint32(stream);
                    byte[] fieldData = new byte[len];
                    stream.Read(fieldData, 0, len);
                    string str = Encoding.UTF8.GetString(fieldData);
                    
                    if (fieldNumber == StringStringEntryProto_Key)
                        key = str;
                    else if (fieldNumber == StringStringEntryProto_Value)
                        value = str;
                }
                else
                {
                    SkipField(stream, wireType);
                }
            }
            
            if (key != null && value != null)
            {
                switch (key)
                {
                    case "location":
                        location = value;
                        break;
                    case "offset":
                        long.TryParse(value, out offset);
                        break;
                    case "length":
                        long.TryParse(value, out length);
                        break;
                }
            }
        }

        private static byte[]? LoadExternalData(string modelDir, string location, long offset, long length)
        {
            try
            {
                string filePath = Path.Combine(modelDir, location);
                if (!File.Exists(filePath))
                {
                    Debug.LogWarning($"[OnnxExternalDataMerger] External data file not found: {filePath}");
                    return null;
                }
                
                using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                
                if (offset > 0)
                    fs.Seek(offset, SeekOrigin.Begin);
                
                int readLength = length > 0 ? (int)length : (int)(fs.Length - offset);
                byte[] data = new byte[readLength];
                int bytesRead = fs.Read(data, 0, readLength);
                
                if (bytesRead != readLength)
                {
                    Debug.LogWarning($"[OnnxExternalDataMerger] Only read {bytesRead} of {readLength} bytes");
                }
                
                return data;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OnnxExternalDataMerger] Error loading external data: {ex.Message}");
                return null;
            }
        }

        private static int ReadVarint32(Stream stream)
        {
            int result = 0;
            int shift = 0;
            int b;
            
            while ((b = stream.ReadByte()) >= 0)
            {
                result |= (b & 0x7F) << shift;
                if ((b & 0x80) == 0)
                    return result;
                shift += 7;
                if (shift >= 32)
                    throw new InvalidDataException("Malformed varint");
            }
            
            return result;
        }

        private static long ReadVarint64(Stream stream)
        {
            long result = 0;
            int shift = 0;
            int b;
            
            while ((b = stream.ReadByte()) >= 0)
            {
                result |= (long)(b & 0x7F) << shift;
                if ((b & 0x80) == 0)
                    return result;
                shift += 7;
                if (shift >= 64)
                    throw new InvalidDataException("Malformed varint");
            }
            
            return result;
        }

        private static void WriteVarint32(Stream stream, int value)
        {
            uint v = (uint)value;
            while (v >= 0x80)
            {
                stream.WriteByte((byte)(v | 0x80));
                v >>= 7;
            }
            stream.WriteByte((byte)v);
        }

        private static void WriteVarint64(Stream stream, long value)
        {
            ulong v = (ulong)value;
            while (v >= 0x80)
            {
                stream.WriteByte((byte)(v | 0x80));
                v >>= 7;
            }
            stream.WriteByte((byte)v);
        }

        private static void SkipField(Stream stream, int wireType)
        {
            switch (wireType)
            {
                case 0: ReadVarint64(stream); break;
                case 1: stream.Seek(8, SeekOrigin.Current); break;
                case 2: stream.Seek(ReadVarint32(stream), SeekOrigin.Current); break;
                case 5: stream.Seek(4, SeekOrigin.Current); break;
                default: throw new InvalidDataException($"Unknown wire type: {wireType}");
            }
        }
    }
}
#endif

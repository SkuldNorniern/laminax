//! Common type mappings and conversions.

use laminax::DType;
use crate::CodegenError;

/// Map Laminax DType to various backend type representations
pub trait TypeMapper {
    /// Get the type name for this backend
    fn map_type(&self, dtype: DType) -> std::result::Result<String, CodegenError>;

    /// Get vector type name for this backend
    fn map_vector_type(&self, dtype: DType, width: usize) -> std::result::Result<String, CodegenError>;

    /// Get pointer type name for this backend
    fn map_pointer_type(&self, dtype: DType) -> std::result::Result<String, CodegenError>;
}

/// C-style type mapper (for CUDA, OpenCL, C/C++)
///
/// Note: When using int64_t/uint64_t types in generated code,
/// ensure #include <stdint.h> is added to the generated headers.
pub struct CTypeMapper;

impl TypeMapper for CTypeMapper {
    fn map_type(&self, dtype: DType) -> std::result::Result<String, CodegenError> {
        Ok(match dtype {
            DType::F32 => "float".to_string(),
            DType::F64 => "double".to_string(),
            DType::I8 => "int8_t".to_string(),
            DType::I16 => "short".to_string(),
            DType::I32 => "int".to_string(),
            DType::I64 => "int64_t".to_string(),
            DType::U8 => "uint8_t".to_string(),
            DType::U16 => "unsigned short".to_string(),
            DType::U32 => "unsigned int".to_string(),
            DType::U64 => "uint64_t".to_string(),
            DType::Bool => "bool".to_string(),
            DType::F16 => "half".to_string(), // CUDA/OpenCL extension
            DType::BF16 => "half".to_string(), // Approximate mapping
            DType::QI4 => "char".to_string(), // Packed format
            DType::QU8 => "unsigned char".to_string(),
        })
    }

    fn map_vector_type(&self, dtype: DType, width: usize) -> std::result::Result<String, CodegenError> {
        Ok(format!("{}{}", self.map_type(dtype)?, width))
    }

    fn map_pointer_type(&self, dtype: DType) -> std::result::Result<String, CodegenError> {
        Ok(format!("{}*", self.map_type(dtype)?))
    }
}

/// Metal Shading Language type mapper
pub struct MetalTypeMapper;

impl TypeMapper for MetalTypeMapper {
    fn map_type(&self, dtype: DType) -> std::result::Result<String, CodegenError> {
        match dtype {
            DType::F32 => Ok("float".to_string()),
            DType::F64 => Ok("double".to_string()),
            DType::I8 => Ok("char".to_string()),
            DType::I16 => Ok("short".to_string()),
            DType::I32 => Ok("int".to_string()),
            DType::I64 => Ok("long".to_string()),
            DType::U8 => Ok("uchar".to_string()),
            DType::U16 => Ok("ushort".to_string()),
            DType::U32 => Ok("uint".to_string()),
            DType::U64 => Ok("ulong".to_string()),
            DType::Bool => Ok("bool".to_string()),
            DType::F16 => Ok("half".to_string()),
            DType::BF16 => Err(crate::CodegenError::UnsupportedType {
                backend: "Metal",
                dtype: DType::BF16,
                reason: "BF16 has FP32-like exponent range with reduced precision, while Metal's 'half' is IEEE 754 FP16 with different exponent/mantissa split. Use F16 instead.",
            }),
            DType::QI4 => Ok("char".to_string()),
            DType::QU8 => Ok("uchar".to_string()),
        }
    }

    fn map_vector_type(&self, dtype: DType, width: usize) -> std::result::Result<String, CodegenError> {
        Ok(format!("{}{}", self.map_type(dtype)?, width))
    }

    fn map_pointer_type(&self, dtype: DType) -> std::result::Result<String, CodegenError> {
        Ok(format!("{}*", self.map_type(dtype)?))
    }
}

/// WGSL (WebGPU Shading Language) type mapper
pub struct WgslTypeMapper;

impl TypeMapper for WgslTypeMapper {
    fn map_type(&self, dtype: DType) -> std::result::Result<String, CodegenError> {
        Ok(match dtype {
            DType::F32 => "f32".to_string(),
            DType::F64 => "f64".to_string(),
            DType::I8 => "i8".to_string(),
            DType::I16 => "i16".to_string(),
            DType::I32 => "i32".to_string(),
            DType::I64 => "i64".to_string(),
            DType::U8 => "u8".to_string(),
            DType::U16 => "u16".to_string(),
            DType::U32 => "u32".to_string(),
            DType::U64 => "u64".to_string(),
            DType::Bool => "bool".to_string(),
            DType::F16 => "f16".to_string(),
            DType::BF16 => "f16".to_string(), // Approximate mapping
            DType::QI4 => "i8".to_string(),
            DType::QU8 => "u8".to_string(),
        })
    }

    fn map_vector_type(&self, dtype: DType, width: usize) -> std::result::Result<String, CodegenError> {
        Ok(format!("vec{}<{}>", width, self.map_type(dtype)?))
    }

    fn map_pointer_type(&self, dtype: DType) -> std::result::Result<String, CodegenError> {
        Ok(format!("&{}", self.map_type(dtype)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CodegenError;

    #[test]
    fn test_ctype_mapper_supports_all_types() {
        let mapper = CTypeMapper;
        // Test that all types are supported and return Ok
        assert!(mapper.map_type(DType::F32).is_ok());
        assert!(mapper.map_type(DType::F64).is_ok());
        assert!(mapper.map_type(DType::BF16).is_ok());
        assert!(mapper.map_type(DType::QI4).is_ok());
    }

    #[test]
    fn test_metal_mapper_bf16_unsupported() {
        let mapper = MetalTypeMapper;
        // Test that BF16 returns UnsupportedType error
        let result = mapper.map_type(DType::BF16);
        assert!(result.is_err());

        if let Err(CodegenError::UnsupportedType { backend, dtype, reason }) = result {
            assert_eq!(backend, "Metal");
            assert_eq!(dtype, DType::BF16);
            assert!(reason.contains("BF16"));
            assert!(reason.contains("IEEE 754 FP16"));
        } else {
            panic!("Expected UnsupportedType error, got {:?}", result);
        }

        // Test that other types work
        assert!(mapper.map_type(DType::F32).is_ok());
        assert!(mapper.map_type(DType::F16).is_ok());
    }

    #[test]
    fn test_wgsl_mapper_supports_all_types() {
        let mapper = WgslTypeMapper;
        // Test that all types are supported
        assert!(mapper.map_type(DType::F32).is_ok());
        assert!(mapper.map_type(DType::F64).is_ok());
        assert!(mapper.map_type(DType::BF16).is_ok());
        assert!(mapper.map_type(DType::QI4).is_ok());
    }

    #[test]
    fn test_vector_type_propagates_errors() {
        let metal_mapper = MetalTypeMapper;

        // BF16 should fail for vector types too
        let result = metal_mapper.map_vector_type(DType::BF16, 4);
        assert!(result.is_err());

        if let Err(CodegenError::UnsupportedType { backend, dtype, .. }) = result {
            assert_eq!(backend, "Metal");
            assert_eq!(dtype, DType::BF16);
        } else {
            panic!("Expected UnsupportedType error, got {:?}", result);
        }

        // Supported types should work
        assert!(metal_mapper.map_vector_type(DType::F32, 4).is_ok());
    }

    #[test]
    fn test_pointer_type_propagates_errors() {
        let metal_mapper = MetalTypeMapper;

        // BF16 should fail for pointer types too
        let result = metal_mapper.map_pointer_type(DType::BF16);
        assert!(result.is_err());

        if let Err(CodegenError::UnsupportedType { backend, dtype, .. }) = result {
            assert_eq!(backend, "Metal");
            assert_eq!(dtype, DType::BF16);
        } else {
            panic!("Expected UnsupportedType error, got {:?}", result);
        }

        // Supported types should work
        assert!(metal_mapper.map_pointer_type(DType::F32).is_ok());
    }
}


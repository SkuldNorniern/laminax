//! Common utility functions for code generation.

/// Sanitize identifier names for various backends
pub fn sanitize_identifier(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            result.push(ch);
        } else {
            result.push('_');
        }
    }
    if result.is_empty() {
        "var".to_string()
    } else if result.chars().next().unwrap().is_ascii_digit() {
        format!("var_{}", result)
    } else {
        result
    }
}

/// Generate a unique name with a numeric suffix
pub fn unique_name(base: &str, counter: &mut usize) -> String {
    let name = format!("{}_{}", base, *counter);
    *counter += 1;
    name
}

/// Check if a data type requires special handling (e.g., packed formats)
pub fn is_packed_type(dtype: laminax::DType) -> bool {
    matches!(dtype, laminax::DType::QI4 | laminax::DType::QU8)
}

/// Get the alignment requirement for a data type (in bytes)
pub fn type_alignment(dtype: laminax::DType) -> usize {
    match dtype {
        laminax::DType::F64 | laminax::DType::I64 | laminax::DType::U64 => 8,
        laminax::DType::F32 | laminax::DType::I32 | laminax::DType::U32 => 4,
        laminax::DType::F16 | laminax::DType::BF16 | laminax::DType::I16 | laminax::DType::U16 => 2,
        _ => 1, // F8, I8, U8, Bool, QI4, QU8
    }
}

/// Check if a backend supports vector operations
pub fn supports_vectors(backend: &str) -> bool {
    matches!(backend, "cuda" | "opencl" | "metal" | "vulkan" | "webgpu")
}

/// Generate appropriate indentation for code generation
pub fn indent(level: usize) -> String {
    "    ".repeat(level)
}

/// Format a function signature for a specific backend
pub fn format_function_signature(
    backend: &str,
    name: &str,
    params: &[(&str, &str)],
    return_type: &str,
) -> String {
    match backend {
        "cuda" => format!(
            "__global__ {} {}({})",
            return_type,
            name,
            params
                .iter()
                .map(|(ty, name)| format!("{} {}", ty, name))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        "opencl" => format!(
            "__kernel {} {}({})",
            return_type,
            name,
            params
                .iter()
                .map(|(ty, name)| format!("__global {} {}", ty, name))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        "metal" => format!(
            "kernel {} {}({})",
            return_type,
            name,
            params
                .iter()
                .enumerate()
                .map(|(i, (ty, name))| format!("device {} {} [[buffer({})]]", ty, name, i))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => format!(
            "{} {}({})",
            return_type,
            name,
            params
                .iter()
                .map(|(ty, name)| format!("{} {}", ty, name))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

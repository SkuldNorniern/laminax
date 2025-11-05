use laminax::lcir::{self, access, index, BinaryOp, KernelBuilder, MemoryScope};
use laminax::{Shape, F32};

fn main() {
    // Build a simple LCIR kernel: C = A + B element-wise for 4 elements
    let mut builder = KernelBuilder::new("vector_add");
    let shape = Shape::from([4usize]);

    let lhs = builder.add_tensor("lhs", shape.clone(), F32, MemoryScope::Global);
    let rhs = builder.add_tensor("rhs", shape.clone(), F32, MemoryScope::Global);
    let out = builder.add_tensor("out", shape.clone(), F32, MemoryScope::Global);

    let loop_id = builder.add_loop("i", 0, shape.dims()[0] as i64, 1);
    let idx = vec![index::loop_var(loop_id)];

    builder.add_binary_op(
        access::tensor(out, idx.clone(), MemoryScope::Global),
        access::tensor(lhs, idx.clone(), MemoryScope::Global),
        BinaryOp::Add,
        access::tensor(rhs, idx, MemoryScope::Global),
    );

    let kernel = builder.build();

    match laminax_codegen::compile_from_lcir(&kernel, laminax_codegen::Backend::Cpu) {
        Ok(asm) => {
            println!("-- Host CPU Assembly (first 20 lines) --");
            for line in String::from_utf8_lossy(&asm).lines().take(20) {
                println!("{}", line);
            }
        }
        Err(err) => {
            eprintln!("Codegen error: {:?}", err);
        }
    }
}

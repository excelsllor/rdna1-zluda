use crate::utils::{Context, Message, PassBuilderOptions, TargetMachine};
use crate::LLVMZludaParseCommandLineOptions;
use crate::{ffi::LLVMZludaLinkWithLLD, utils::Module};
use llvm_sys::{
    core::*,
    target::{
        LLVMInitializeAMDGPUAsmPrinter, LLVMInitializeAMDGPUTarget, LLVMInitializeAMDGPUTargetInfo,
        LLVMInitializeAMDGPUTargetMC,
    },
    target_machine::{
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMGetTargetFromTriple,
        LLVMRelocMode,
    },
    transforms::pass_builder::LLVMRunPasses,
};
use llvm_sys::{LLVMLinkage, LLVMVisibility};
use std::ffi::{CStr, CString};
use std::sync::OnceLock;
use std::{fs, ptr};
use tempfile::NamedTempFile;

const OCKL_MODULE: &[u8] = include_bytes!("device-libs/ockl.bc");
const LD_MATRIX_SHIMS: &[u8] = br#"; ModuleID = 'zluda_shims'
source_filename = "zluda_shims.ll"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)

define linkonce_odr <2 x i32> @__zluda_ptx_impl_ldmatrix_m8n8_x2_trans_b16(ptr addrspace(3) %address) {
entry:
  %lane_lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane_lo)
  %x0 = call i32 @zluda_ldmatrix_trans_single(ptr addrspace(3) %address, i32 0, i32 %lane)
  %x1 = call i32 @zluda_ldmatrix_trans_single(ptr addrspace(3) %address, i32 8, i32 %lane)
  %v0 = insertelement <2 x i32> poison, i32 %x0, i32 0
  %v1 = insertelement <2 x i32> %v0, i32 %x1, i32 1
  ret <2 x i32> %v1
}

define internal i32 @zluda_ldmatrix_trans_single(ptr addrspace(3) %address, i32 %warp_offset, i32 %lane) {
entry:
  %base = ptrtoint ptr addrspace(3) %address to i32
  %lane_mod4 = and i32 %lane, 3
  %lane_div4 = lshr i32 %lane, 2
  %lane_div4_mul2 = shl i32 %lane_div4, 1
  %lane_mod4_mul2 = shl i32 %lane_mod4, 1
  %off_lo = add i32 %lane_mod4_mul2, %warp_offset
  %byte_lo = shl i32 %off_lo, 2
  %ra_lo = call i32 @llvm.amdgcn.ds.bpermute(i32 %byte_lo, i32 %base)
  %a_lo = add i32 %ra_lo, %lane_div4_mul2
  %p_lo = inttoptr i32 %a_lo to ptr addrspace(3)
  %lo = load i16, ptr addrspace(3) %p_lo, align 2
  %off_hi = add i32 %off_lo, 1
  %byte_hi = shl i32 %off_hi, 2
  %ra_hi = call i32 @llvm.amdgcn.ds.bpermute(i32 %byte_hi, i32 %base)
  %a_hi = add i32 %ra_hi, %lane_div4_mul2
  %p_hi = inttoptr i32 %a_hi to ptr addrspace(3)
  %hi = load i16, ptr addrspace(3) %p_hi, align 2
  %vec0 = insertelement <2 x i16> poison, i16 %lo, i32 0
  %vec1 = insertelement <2 x i16> %vec0, i16 %hi, i32 1
  %packed = bitcast <2 x i16> %vec1 to i32
  ret i32 %packed
}
"#;


// https://llvm.org/docs/AMDGPUUsage.html#address-spaces
const CONSTANT_ADDRESS_SPACE: u32 = 4;

fn load_module(ctx: &Context, bc: &[u8], name: &std::ffi::CStr) -> Result<Module, String> {
    let module = Module::try_from_bitcode(ctx, bc, Some(name))
        .ok_or(("Failed to parse bitcode").to_string())?;
    module.verify()?;
    Ok(module)
}

// TODO: see if there's a way to reduce duplication with attributes.rs
fn add_constant(context: &Context, module: &Module, name: &CStr, attribute: u32) {
    let attribute_type = unsafe { LLVMInt32TypeInContext(context.get()) };
    let global = unsafe {
        LLVMAddGlobalInAddressSpace(
            module.get(),
            attribute_type,
            name.as_ptr(),
            CONSTANT_ADDRESS_SPACE,
        )
    };
    unsafe { LLVMSetLinkage(global, LLVMLinkage::LLVMExternalLinkage) };
    unsafe { LLVMSetVisibility(global, LLVMVisibility::LLVMHiddenVisibility) };
    unsafe { LLVMSetInitializer(global, LLVMConstInt(attribute_type, attribute as u64, 0)) };
    unsafe { LLVMSetGlobalConstant(global, 1) };
}

fn path_to_cstring(path: &std::path::Path) -> Result<CString, String> {
    let path_str = path
        .to_str()
        .ok_or(("path is not valid as str").to_string())?;
    CString::new(path_str).map_err(|_| ("path includes invalid null byte").to_string())
}

fn get_isa_version_from_gcn_arch(gcn_arch: &str) -> Result<u32, String> {
    // ROCm reports the architecture together with target features, e.g.
    // "gfx1010:xnack-". Only the leading "gfxNNNN" token is numeric.
    let gcn_arch = gcn_arch.split(':').next().unwrap_or(gcn_arch);
    let base: u32 = gcn_arch
        .replace("gfx", "")
        .parse()
        .map_err(|_| ("could not get ISA version from gcn_arch").to_string())?;
    let stepping = base % 10;
    let minor = (base / 10) % 10;
    let major = base / 100;
    Ok(major * 1000 + minor * 100 + stepping)
}

fn create_oclc_constants(ctx: &Context, gcn_arch: &str) -> Result<Module, String> {
    let module = Module::new(ctx, c"oclc_constants");

    // used by ockl
    add_constant(ctx, &module, c"__oclc_wavefrontsize64", 0);
    add_constant(ctx, &module, c"__oclc_wavefrontsize_log2", 5);
    add_constant(ctx, &module, c"__oclc_ABI_version", 500);
    add_constant(
        ctx,
        &module,
        c"__oclc_ISA_version",
        get_isa_version_from_gcn_arch(gcn_arch)?,
    );

    // used by ocml
    add_constant(ctx, &module, c"__oclc_unsafe_math_opt", 0);
    add_constant(ctx, &module, c"__oclc_correctly_rounded_sqrt32", 1);
    add_constant(ctx, &module, c"__oclc_finite_only_opt", 0);
    Ok(module)
}

pub fn compile(
    ctx: &Context,
    gcn_arch: &str,
    main: Module,
    ptx_impl: &[u8],
    attributes: Module,
    metadata: kernel_metadata::ModuleMetadataV1,
    metadata32: Option<kernel_metadata::ModuleMetadata32Bit>,
    compiler_hook: Option<&dyn Fn(&Vec<u8>, String)>,
) -> Result<Vec<u8>, String> {
    init_globals()?;

    let linked = Module::new(ctx, c"llvm-link");

    let ptx_impl = load_module(ctx, ptx_impl, c"ptx_impl.bc")?;
    let ockl = load_module(ctx, OCKL_MODULE, c"ockl.bc")?;

    let oclc_constants = create_oclc_constants(ctx, gcn_arch)?;

    linked.link(main)?;
    linked.link(attributes)?;
    linked.link(oclc_constants)?;
    linked.link(ptx_impl)?;
    linked.link(ockl)?;

    // Textual IR cannot be parsed into a context that discards value names.
    unsafe { LLVMContextSetDiscardValueNames(ctx.get(), 0) };
    let shims = Module::try_from_ir(ctx, LD_MATRIX_SHIMS, Some(c"zluda_shims.ll"));
    unsafe { LLVMContextSetDiscardValueNames(ctx.get(), 1) };
    let shims = shims.ok_or(("Failed to parse shim IR").to_string())?;
    unsafe {
        LLVMSetDataLayout(shims.get(), LLVMGetDataLayoutStr(linked.get()));
        LLVMSetTarget(shims.get(), LLVMGetTarget(linked.get()));
    }
    linked.link(shims)?;

    linked.verify()?;

    if let Some(hook) = compiler_hook {
        // Run compiler hook on human-readable LLVM IR
        let message = linked.print_module_to_string();
        let data = message.to_bytes().to_vec();
        hook(&data, String::from("linked.ll"));
    }

    let triple = c"amdgcn-amd-amdhsa";
    // ROCm reports architecture with target features, e.g. "gfx1010:xnack-",
    // but the AMDGPU TargetMachine only accepts the bare processor name.
    let cpu_name = gcn_arch.split(':').next().unwrap_or(gcn_arch);
    let cpu = std::ffi::CString::new(cpu_name).map_err(|_| ("invalid gcn_arch").to_string())?;
    let features = c"-wavefrontsize64,+cumode";

    let mut target = unsafe { std::mem::zeroed() };
    let mut err = std::ptr::null_mut();
    let status = unsafe { LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, &mut err) };
    if status != 0 {
        let message = Message::new(unsafe { CStr::from_ptr(err) });
        return Err(message.to_str().to_string());
    }

    let target_machine = TargetMachine::new(
        target,
        triple,
        &cpu,
        features,
        LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        LLVMRelocMode::LLVMRelocDefault,
        LLVMCodeModel::LLVMCodeModelDefault,
    );

    let pb_options = PassBuilderOptions::new();
    let error = unsafe {
        LLVMRunPasses(
            linked.get(),
            c"default<O3>".as_ptr(),
            target_machine.get(),
            pb_options.get(),
        )
    };

    if !error.is_null() {
        let err_msg = unsafe { llvm_sys::error::LLVMGetErrorMessage(error) };
        let message = Message::new(unsafe { CStr::from_ptr(err_msg) });
        return Err(message.to_str().to_string());
    }

    if let Some(hook) = compiler_hook {
        // Run compiler hook on optimized human-readable LLVM IR
        let message = linked.print_module_to_string();
        let data = message.to_bytes().to_vec();
        hook(&data, String::from("opt.ll"));

        // Running a disassembler would be a bit of a pain, so run codegen as assembly
        let assembly = target_machine
            .emit_to_memory_buffer(&linked.clone(), LLVMCodeGenFileType::LLVMAssemblyFile)?
            .to_vec();
        hook(&assembly, String::from("asm"))
    }

    let object_file = target_machine
        .emit_to_memory_buffer(&linked, LLVMCodeGenFileType::LLVMObjectFile)?
        .to_vec();

    if let Some(hook) = compiler_hook {
        // Run compiler hook for object file
        hook(&object_file, String::from("o"));
    }

    let object_path = NamedTempFile::with_prefix("zluda.o")
        .map_err(|e| format!("Failed to create temporary file: {}", e))?
        .into_temp_path();
    let mut section_path = NamedTempFile::with_prefix("zluda_section")
        .map_err(|e| format!("Failed to create temporary file: {}", e))?;
    let executable_path = NamedTempFile::with_prefix("zluda.elf")
        .map_err(|e| format!("Failed to create temporary file: {}", e))?
        .into_temp_path();

    fs::write(&object_path, &object_file)
        .map_err(|e| format!("Failed to write object file: {}", e))?;
    metadata
        .write_object(&object_file, section_path.as_file_mut())
        .map_err(|e| format!("Failed to write metadata section: {}", e))?;
    let section32_path = if let Some(metadata32) = metadata32 {
        let mut section32_path = NamedTempFile::with_prefix("zluda32_section")
            .map_err(|e| format!("Failed to create temporary file: {}", e))?;
        metadata32
            .write_object(&object_file, section32_path.as_file_mut())
            .map_err(|e| format!("Failed to write 32-bit metadata section: {}", e))?;
        Some(section32_path)
    } else {
        None
    };

    let object_path_cstr = path_to_cstring(&object_path)?;
    let section_path_cstr = path_to_cstring(section_path.as_ref())?;
    let section32_path_cstr = section32_path
        .as_ref()
        .map(|p| path_to_cstring(p.as_ref()))
        .transpose()?;
    let executable_path_cstr = path_to_cstring(&executable_path)?;

    let (inputs, inputs_len) = if let Some(ref section32_path_cstr) = section32_path_cstr {
        (
            [
                object_path_cstr.as_ptr(),
                section_path_cstr.as_ptr(),
                section32_path_cstr.as_ptr(),
            ],
            3,
        )
    } else {
        (
            [
                object_path_cstr.as_ptr(),
                section_path_cstr.as_ptr(),
                ptr::null(),
            ],
            2,
        )
    };

    let mut err = std::ptr::null_mut();
    let result = unsafe {
        LLVMZludaLinkWithLLD(
            inputs_len,
            inputs.as_ptr(),
            executable_path_cstr.as_ptr(),
            &mut err,
        )
    };
    if result != 0 {
        let message = Message::new(unsafe { CStr::from_ptr(err) });
        return Err(message.to_str().to_string());
    }

    let executable =
        fs::read(&executable_path).map_err(|_| ("Failed to read executable file").to_string())?;

    if let Some(hook) = compiler_hook {
        // Run compiler hook for final executable
        hook(&executable, String::from("elf"));
    }

    Ok(executable)
}

fn init_globals() -> Result<(), String> {
    static INIT_AMDGPU: OnceLock<Result<(), Message>> = OnceLock::new();
    INIT_AMDGPU
        .get_or_init(|| {
            let common_options = vec![
                // Uncomment for LLVM debug
                //c"-debug",
                // Uncomment to save passes
                // c"-print-before-all",
                c"llvm_zluda",
                //c"-debug-only=isel",
                c"-ignore-tti-inline-compatible",
                // c"-amdgpu-early-inline-all=true",
                c"-amdgpu-internalize-symbols",
                c"-amdhsa-code-object-version=5",
                //c"--pass-remarks-missed=.*inlin.*",
            ]
            .into_iter();
            /* Does not provide as much performance improvement on ROCm 7.2
            let opt_options = if cfg!(debug_assertions) {
                vec![]
            } else {
                vec![
                    // default inlining threshold times 10
                    c"-inline-threshold=2250",
                    c"-inlinehint-threshold=3250",
                ]
            };
            */
            let llvm_args_ptrs: Vec<*const i8> = common_options
                //.chain(opt_options)
                .map(|s| s.as_ptr())
                .collect();
            let mut err_msg = std::ptr::null_mut();
            let success = unsafe {
                LLVMZludaParseCommandLineOptions(
                    llvm_args_ptrs.len() as i32,
                    llvm_args_ptrs.as_ptr(),
                    &mut err_msg,
                )
            };
            if !success {
                return Err(Message::new(unsafe { CStr::from_ptr(err_msg) }));
            }
            unsafe { LLVMInitializeAMDGPUTargetInfo() };
            unsafe { LLVMInitializeAMDGPUTarget() };
            unsafe { LLVMInitializeAMDGPUTargetMC() };
            unsafe { LLVMInitializeAMDGPUAsmPrinter() };
            Ok(())
        })
        .as_ref()
        .map(|()| ())
        .map_err(|e| e.to_str().to_string())
}

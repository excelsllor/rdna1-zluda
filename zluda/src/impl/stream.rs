use hip_runtime_sys::*;

pub(crate) fn synchronize(stream: hipStream_t) -> hipError_t {
    unsafe { hipStreamSynchronize(stream) }
}

pub(crate) fn create(stream: *mut hipStream_t, flags: ::core::ffi::c_uint) -> hipError_t {
    unsafe { hipStreamCreateWithFlags(stream, flags) }
}

pub(crate) fn create_with_priority(
    stream: *mut hipStream_t,
    flags: ::core::ffi::c_uint,
    priority: ::core::ffi::c_int,
) -> hipError_t {
    unsafe { hipStreamCreateWithPriority(stream, flags, priority) }
}

pub(crate) fn destroy_v2(stream: hipStream_t) -> hipError_t {
    unsafe { hipStreamDestroy(stream) }
}

pub(crate) fn begin_capture(stream: hipStream_t) -> hipError_t { eprintln!("zluda: cuStreamBeginCapture called");
    // legacy (non-v2) cuStreamBeginCapture has no mode argument; CUDA uses
    // CU_STREAM_CAPTURE_MODE_GLOBAL semantics for it.
    unsafe { hipStreamBeginCapture(stream, hipStreamCaptureMode(0)) }
}
pub(crate) fn begin_capture_v2(stream: hipStream_t, mode: hipStreamCaptureMode) -> hipError_t { eprintln!("zluda: cuStreamBeginCapture_v2 called");
    unsafe { hipStreamBeginCapture(stream, mode) }
}

pub(crate) fn end_capture(stream: hipStream_t, graph: *mut hipGraph_t) -> hipError_t { eprintln!("zluda: cuStreamEndCapture called");
    unsafe { hipStreamEndCapture(stream, graph) }
}

pub(crate) fn is_capturing(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
) -> hipError_t {
    unsafe { hipStreamIsCapturing(stream, capture_status) }
}

pub(crate) fn get_capture_info(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
    id: *mut ::core::ffi::c_ulonglong,
) -> hipError_t {
    unsafe { hipStreamGetCaptureInfo(stream, capture_status, id) }
}
pub(crate) fn get_capture_info_v2(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
    id: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hipGraph_t,
    dependencies_out: *mut *const hipGraphNode_t,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    unsafe {
        hipStreamGetCaptureInfo_v2(
            stream,
            capture_status,
            id,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        )
    }
}

pub(crate) fn get_capture_info_v3(
    stream: hipStream_t,
    capture_status_out: *mut hipStreamCaptureStatus,
    id_out: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hipGraph_t,
    dependencies_out: *mut *const hipGraphNode_t,
    edge_data_out: *mut *const cuda_types::cuda::CUgraphEdgeData,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    if !edge_data_out.is_null() {
        return hipError_t::ErrorNotSupported;
    }
    unsafe {
        hipStreamGetCaptureInfo_v2(
            stream,
            capture_status_out,
            id_out,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        )
    }
}

pub(crate) fn wait_event(
    stream: hipStream_t,
    event: hipEvent_t,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    unsafe { hipStreamWaitEvent(stream, event, flags) }
}

pub(crate) fn query(stream: hipStream_t) -> hipError_t {
    unsafe { hipStreamQuery(stream) }
}

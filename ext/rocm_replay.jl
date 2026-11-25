using AMDGPU: GPUArrays

DianoiaML.replay_warmups(::ROCBackend) = 2
DianoiaML.replay_cache(::ROCBackend) = GPUArrays.AllocCache()

function DianoiaML.warm_replay(f, ::ROCBackend, cache, warmed)
    # The first call creates persistent layer buffers. The second caches
    # temporary allocations so capture can reuse their storage.
    return warmed == 0 ? f() : GPUArrays.@cached(cache, f())
end

function DianoiaML.capture_replay(f, ::ROCBackend, cache)
    hip = AMDGPU.HIP
    executable = try
        graph = GPUArrays.@cached cache hip.capture(f; flags=hip.hipStreamCaptureModeRelaxed, throw_error=false)
        graph === nothing && return nothing
        hip.instantiate(graph)
    catch error
        if error isa hip.HIPError &&
           error.code in (hip.hipErrorNotSupported, hip.hipErrorStreamCaptureUnsupported,
                          hip.hipErrorStreamCaptureInvalidated, hip.hipErrorStreamCaptureImplicit,
                          hip.hipErrorCapturedEvent)
            return nothing
        end
        rethrow()
    end
    return () -> GC.@preserve cache hip.launch(executable)
end

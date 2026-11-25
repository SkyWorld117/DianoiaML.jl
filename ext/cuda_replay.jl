# Use the capture API that keeps captured device allocations alive.
DianoiaML.replay_warmups(::CUDABackend) = pkgversion(CUDACore) >= v"6.5" ? 1 : 0

function DianoiaML.capture_replay(f, ::CUDABackend, ::Nothing)
    graph = CUDACore.capture(f; throw_error=false)
    graph === nothing && return nothing
    executable = CUDACore.instantiate(graph)
    return () -> (executable(); nothing)
end

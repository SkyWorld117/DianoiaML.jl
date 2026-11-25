const Ze = oneAPI.oneL0

struct GraphAPI
    begin_capture::Ptr{Cvoid}
    end_capture::Ptr{Cvoid}
    instantiate::Ptr{Cvoid}
    append::Ptr{Cvoid}
    destroy_graph::Ptr{Cvoid}
    destroy_executable::Ptr{Cvoid}
end

function graph_api(driver)
    haskey(Ze.extension_properties(driver), "ZE_extension_record_replay_graph") || return nothing
    # Use the loader's entry points so the handles have the same representation
    # as the handles created by oneAPI.jl.
    lib = oneAPI.Libdl.dlopen(Ze.libze_loader; throw_error=false)
    lib === nothing && return nothing
    try
        names = (:zeCommandListBeginGraphCaptureExt, :zeCommandListEndGraphCaptureExt,
                 :zeGraphInstantiateExt, :zeCommandListAppendGraphExt,
                 :zeGraphDestroyExt, :zeExecutableGraphDestroyExt)
        entries = map(name -> oneAPI.Libdl.dlsym(lib, name; throw_error=false), names)
        any(entry -> entry === nothing || entry == C_NULL, entries) && return nothing
        return GraphAPI(entries...)
    finally
        oneAPI.Libdl.dlclose(lib)  # oneAPI.jl already holds the loader open.
    end
end

DianoiaML.replay_warmups(::oneAPIBackend) = 1

struct IntelLaunch{A}
    kernel::Ze.ZeKernel
    groups::Dims{3}
    items::Dims{3}
    args::A
end

function prepare_launch(launch, stream, refs)
    (; host, f) = launch.kernel.kern
    oneAPI.kernel_convert(f, refs)
    args = map(arg -> oneAPI.kernel_convert(arg, refs), launch.args)
    values = (oneAPI.KernelState(stream), host.f, args...)
    arguments = Tuple(isbitstype(typeof(x)) ? x : C_NULL for x in values
                      if !(oneAPI.isghosttype(Core.Typeof(x)) || Core.Compiler.isconstType(Core.Typeof(x))))
    return IntelLaunch(host.fun, launch.groups, launch.items, arguments)
end

function append_launch!(list, launch::IntelLaunch)
    kernel = launch.kernel
    @lock kernel begin
        for (i, arg) in enumerate(launch.args)
            Ze.arguments(kernel)[i] = arg
        end
        Ze.groupsize!(kernel, launch.items)
        Ze.append_launch!(list, kernel, launch.groups)
    end
    return
end

# Level Zero's Julia bindings predate the record/replay extension.
const ListHandle = Ze.ze_command_list_handle_t
const EventHandle = Ze.ze_event_handle_t
const GraphHandle = Ptr{Cvoid}
const Result = Ze.ze_result_t
# Result codes introduced with Level Zero's record/replay extension.
const GRAPH_CAPTURE_UNSUPPORTED = Result(0x78000025)
const GRAPH_CAPTURE_INVALIDATED = Result(0x78000026)

capture_unsupported(code) =
    code in (Ze.RESULT_ERROR_UNSUPPORTED_FEATURE, Ze.RESULT_ERROR_UNSUPPORTED_VERSION,
             Ze.RESULT_ERROR_NOT_AVAILABLE, Ze.RESULT_ERROR_DEPENDENCY_UNAVAILABLE,
             GRAPH_CAPTURE_UNSUPPORTED, GRAPH_CAPTURE_INVALIDATED)

function graph_available(code)
    code == Ze.RESULT_SUCCESS && return true
    capture_unsupported(code) && return false
    Ze.throw_api_error(code)
end

function record_graph!(api, list, launches)
    source = Ref{GraphHandle}(C_NULL)
    code = @ccall $(api.begin_capture)(list::ListHandle, C_NULL::Ptr{Cvoid})::Result
    graph_available(code) || return C_NULL
    complete = false
    try
        foreach(launch -> append_launch!(list, launch), launches)
        complete = true
    catch err
        err isa Ze.ZeError && capture_unsupported(err.code) || rethrow()
    finally
        code = @ccall $(api.end_capture)(list::ListHandle, C_NULL::Ptr{Cvoid},
                                         source::Ref{GraphHandle})::Result
        complete = graph_available(code) && complete
        if !complete && source[] != C_NULL
            @ccall $(api.destroy_graph)(source[]::GraphHandle)::Result
        end
    end
    return complete ? source[] : C_NULL
end

mutable struct IntelReplay
    const api::GraphAPI
    const source::GraphHandle
    const executable::GraphHandle
    const capture_list::Ze.ZeImmediateCommandList
    const stream::oneAPI.oneStream
    const managed::Vector{oneAPI.Managed}
    const launches::Vector{DianoiaML.RecordedLaunch}
end

function release_replay!(replay)
    GC.@preserve replay begin
        Ze.nonblocking_synchronize(replay.stream.immediate)
        api = replay.api
        @ccall $(api.destroy_executable)(replay.executable::GraphHandle)::Result
        @ccall $(api.destroy_graph)(replay.source::GraphHandle)::Result
    end
    return
end

function DianoiaML.capture_replay(f, ::oneAPIBackend, ::Nothing)
    Ze.sync_each_submission() && return nothing
    api = graph_api(oneAPI.context().driver)
    api === nothing && return nothing
    launches = DianoiaML.record_launches(f)
    isempty(launches) && return nothing
    stream = oneAPI.global_stream(oneAPI.context(), oneAPI.device())
    oneAPI.mkl_wait!(stream)
    refs = oneAPI.DataRef[]
    commands = [prepare_launch(launch, stream, refs) for launch in launches]
    spill = maximum(command -> Ze.spill_mem_size(command.kernel), commands)
    spill > stream.scratch_hwm && oneAPI.scratch_hedge!(stream, spill)
    managed = oneAPI.Managed[ref[] for ref in refs]
    # A private list keeps an unsuccessful recording off the user stream.
    # Recording and playback use the same in-order asynchronous configuration.
    list = Ze.ZeImmediateCommandList(stream.ctx, stream.dev, stream.immediate.ordinal;
                                    flags=Ze.ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                                    mode=Ze.ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                    priority=stream.priority)
    source = GC.@preserve launches record_graph!(api, list, commands)
    source == C_NULL && return nothing
    executable = Ref{GraphHandle}(C_NULL)
    try
        result = @ccall $(api.instantiate)(source::GraphHandle, C_NULL::Ptr{Cvoid},
                                       executable::Ref{GraphHandle})::Result
        graph_available(result) || return nothing
    finally
        if executable[] == C_NULL
            @ccall $(api.destroy_graph)(source::GraphHandle)::Result
        end
    end
    replay = IntelReplay(api, source, executable[], list, stream, managed, launches)
    finalizer(replay) do graph
        # Run the wait on a Julia task. Keep graph, kernels and arrays rooted
        # through the last submitted execution.
        schedule(Task(() -> release_replay!(graph)))
    end
    return () -> launch_replay!(replay)
end

function launch_replay!(replay::IntelReplay)
    stream = replay.stream
    stream === oneAPI.global_stream(oneAPI.context(), oneAPI.device()) || return false
    api = replay.api
    status = GC.@preserve replay oneAPI.with_ownership(stream, replay.managed) do
        Ze.execute!(stream) do list
            oneAPI.register_pending!(stream)
            @ccall $(api.append)(list::ListHandle, replay.executable::GraphHandle,
                              C_NULL::Ptr{Cvoid}, C_NULL::EventHandle, 0::UInt32,
                              C_NULL::Ptr{EventHandle})::Result
        end
    end
    return graph_available(status)
end

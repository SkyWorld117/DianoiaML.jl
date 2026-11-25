using Metal.ObjectiveC: @objc, id
using Metal.ObjectiveC.Foundation: NSError, NSRange, NSUInteger

const MTL = Metal.MTL

DianoiaML.backend_replayable(::MetalBackend, model, example, loss) =
    DianoiaML.recordable_workload(model, example, loss)

DianoiaML.replay_warmups(::MetalBackend) = 1

struct IndirectArguments
    command::MTL.MTLIndirectComputeCommand
    pointers::Set{Metal.MtlPtr}
    resources::Set{MTL.MTLBuffer}
    constants::Vector{Metal.MtlArray{UInt8,1,Metal.SharedStorage}}
end

Metal.mtlconvert(arg, args::IndirectArguments, ::Nothing) = Metal.Adapt.adapt(args, arg)

function Metal.Adapt.adapt_storage(args::IndirectArguments, array::Metal.MtlArray)
    ptr = pointer(array)
    push!(args.pointers, ptr)
    push!(args.resources, ptr.buffer)
    return Metal.mtlconvert(array)
end

function Metal.bind_pointer!(args::IndirectArguments, ::Nothing, ptr::Metal.MtlPtr, index::Integer)
    push!(args.pointers, ptr)
    return Metal.bind_buffer!(args, ptr.buffer, ptr.offset, index)
end

function Metal.bind_buffer!(args::IndirectArguments, buffer::MTL.MTLBuffer, offset, index)
    push!(args.resources, buffer)
    command = args.command
    @objc [command::id{MTL.MTLIndirectComputeCommand} setKernelBuffer:buffer::id{MTL.MTLBuffer}
           offset:offset::NSUInteger atIndex:(index - 1)::NSUInteger]::Nothing
    return
end

function Metal.bind_bytes!(args::IndirectArguments, ptr::Ptr, len::Integer, index::Integer)
    # Indirect commands bind buffers for scalar and by-value arguments.
    bytes = Vector{UInt8}(undef, len)
    unsafe_copyto!(pointer(bytes), Ptr{UInt8}(ptr), len)
    storage = Metal.MtlArray{UInt8,1,Metal.SharedStorage}(bytes)
    push!(args.constants, storage)
    return Metal.bind_pointer!(args, nothing, pointer(storage), index)
end

function indirect_buffer(device, count)
    desc = @objc [MTLIndirectCommandBufferDescriptor new]::MTL.MTLIndirectCommandBufferDescriptor
    desc.commandTypes = MTL.MTLIndirectCommandTypeConcurrentDispatch
    desc.inheritPipelineState = false
    desc.inheritBuffers = false
    desc.maxKernelBufferBindCount = 31
    options = MTL.MTLResourceOptions(0)
    return @objc [device::id{MTL.MTLDevice} newIndirectCommandBufferWithDescriptor:desc::id{MTL.MTLIndirectCommandBufferDescriptor}
                  maxCommandCount:count::NSUInteger options:options::MTL.MTLResourceOptions]::Union{Nothing,MTL.MTLIndirectCommandBuffer}
end

# Enable indirect execution when linking the same compiled Julia kernel.
function indirect_pipeline(kernel::Metal.HostKernel{F,S,TT}) where {F,S,TT}
    device = kernel.device
    minthreads = iszero(kernel.reqthreads.width) ? nothing :
                 (Int(kernel.reqthreads.width), Int(kernel.reqthreads.height), Int(kernel.reqthreads.depth))
    config = Metal.compiler_config(device; minthreads)
    results = Metal.compile_or_lookup(Metal.CompilerJob(Metal.methodinstance(F, TT), config))
    library = MTL.MTLLibraryFromData(device, results.metallib)
    desc = MTL.MTLComputePipelineDescriptor()
    desc.computeFunction = MTL.MTLFunction(library, results.entry)
    desc.supportIndirectCommandBuffers = true
    minthreads === nothing || (desc.requiredThreadsPerThreadgroup = MTL.MTLSize(minthreads))
    return MTL.MTLComputePipelineState(device, desc)
end

function encode_launch!(buffer, i, launch, pipelines, pointers, resources, constants)
    host = launch.kernel.kern
    pipeline = get!(() -> indirect_pipeline(host), pipelines, host.pipeline)
    prod(launch.items) <= pipeline.maxTotalThreadsPerThreadgroup || return false
    groups, threads = MTL.MTLSize(launch.groups), MTL.MTLSize(launch.items)
    Metal.check_launch(groups, threads, host.reqthreads, Int(pipeline.maxTotalThreadsPerThreadgroup),
                       Int(pipeline.staticThreadgroupMemoryLength))
    command = @objc [buffer::id{MTL.MTLIndirectCommandBuffer} indirectComputeCommandAtIndex:(i - 1)::NSUInteger]::MTL.MTLIndirectComputeCommand
    @objc [command::id{MTL.MTLIndirectComputeCommand} setComputePipelineState:pipeline::id{MTL.MTLComputePipelineState}]::Nothing
    if i > 1
        @objc [command::id{MTL.MTLIndirectComputeCommand} setBarrier]::Nothing
    end

    scratch, scratch_address = Metal.malloc_buffer_and_gpu_address(host.device)
    exception, exception_address = Metal.exception_info_buffer_and_gpu_address(host.device)
    push!(resources, scratch, exception)
    reloc = host.reloc_table
    reloc_address = reloc === nothing ? UInt64(0) : UInt64(MTL.gpu_address(reloc))
    reloc === nothing || push!(resources, reloc)
    state = Metal.KernelState(UInt32(0), host.exception_table,
                              reinterpret(Core.LLVMPtr{UInt8,Metal.AS.Device}, scratch_address),
                              reinterpret(Core.LLVMPtr{UInt8,Metal.AS.Device}, exception_address),
                              reinterpret(Core.LLVMPtr{UInt64,Metal.AS.Device}, reloc_address))
    args = IndirectArguments(command, pointers, resources, constants)
    Metal.bind_arguments!(args, nothing, host, state, launch.args)
    @objc [command::id{MTL.MTLIndirectComputeCommand} concurrentDispatchThreadgroups:groups::MTL.MTLSize
           threadsPerThreadgroup:threads::MTL.MTLSize]::Nothing
    return true
end

struct MetalReplay
    buffer::MTL.MTLIndirectCommandBuffer
    queue::Metal.ConcreteBatchedQueue
    pointers::Vector{Metal.MtlPtr}
    resources::Vector{MTL.MTLBuffer}
    constants::Vector{Metal.MtlArray{UInt8,1,Metal.SharedStorage}}
    pipelines::Dict{MTL.MTLComputePipelineState,MTL.MTLComputePipelineState}
    launches::Vector{DianoiaML.RecordedLaunch}
end

function DianoiaML.capture_replay(f, ::MetalBackend, ::Nothing)
    device = Metal.device()
    Metal.is_virtual(device) && return nothing
    launches = DianoiaML.record_launches(f)
    0 < length(launches) <= 16384 || return nothing
    any(launch -> launch.kernel.kern.device_rng || launch.kernel.kern.loggingEnabled, launches) && return nothing
    buffer = indirect_buffer(device, length(launches))
    buffer === nothing && return nothing
    pipelines = Dict{MTL.MTLComputePipelineState,MTL.MTLComputePipelineState}()
    pointers, resources = Set{Metal.MtlPtr}(), Set{MTL.MTLBuffer}()
    constants = Metal.MtlArray{UInt8,1,Metal.SharedStorage}[]
    try
        for (i, launch) in enumerate(launches)
            encode_launch!(buffer, i, launch, pipelines, pointers, resources, constants) || return nothing
        end
    catch err
        err isa NSError || rethrow()
        return nothing
    end
    replay = MetalReplay(buffer, Metal.global_queue(device), collect(pointers), collect(resources),
                         constants, pipelines, launches)
    return () -> launch_replay!(replay)
end

function submit_replay!(replay)
    queue = replay.queue
    for ptr in replay.pointers
        conflict = Metal.try_take_ownership!(queue, ptr)
        conflict === nothing || return conflict
    end
    Metal.commit_batch!(queue)
    Metal.order_external!(queue)
    cmdbuf = MTL.MTLCommandBuffer(queue.queue)
    MTL.MTLComputeCommandEncoder(cmdbuf; dispatch_type=MTL.MTLDispatchTypeConcurrent) do encoder
        MTL.use!(encoder, replay.resources)
        buffer = replay.buffer
        range = NSRange(0, length(replay.launches))
        @objc [encoder::id{MTL.MTLComputeCommandEncoder} executeCommandsInBuffer:buffer::id{MTL.MTLIndirectCommandBuffer}
               withRange:range::NSRange]::Nothing
    end
    MTL.commit_with_queue_key!(cmdbuf, pointer(queue.queue))
    Metal.defer_cleanup!(queue, cmdbuf, Any[replay])
    return nothing
end

function launch_replay!(replay::MetalReplay)
    replay.queue === Metal.global_queue(Metal.device()) || return false
    while true
        conflict = @lock Metal.submission_lock submit_replay!(replay)
        conflict === nothing && break
        Metal.synchronize(conflict)
    end
    Metal.limit_inflight!(replay.queue)
    return true
end

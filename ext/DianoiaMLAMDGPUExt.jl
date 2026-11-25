module DianoiaMLAMDGPUExt

using AMDGPU: ROCBackend
import DianoiaML

DianoiaML.gpu_backend(::Val{:amd}) = ROCBackend()

end

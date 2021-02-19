module AVX_acc
    using LoopVectorization
    export avx_mul!

    function avx_mul!(C, A, B)
        @avx for x in axes(A, 1), y in axes(B, 2)
            for z in axes(A, 2)
                C[x,y] += A[x,z]*B[z, y]
            end
        end
    end
end

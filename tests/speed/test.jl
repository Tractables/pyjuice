using ProbabilisticCircuits
using MLDatasets
using CUDA


mnist_train_cpu = collect(transpose(reshape(MNIST.traintensor(UInt8), 28*28, :)))
mnist_test_cpu = collect(transpose(reshape(MNIST.testtensor(UInt8), 28*28, :)))
mnist_train_gpu = cu(mnist_train_cpu)
mnist_test_gpu = cu(mnist_test_cpu)


bits = 4
latents = 64
println("Generating HCLT structure with $latents latents... ");
trunc_train = cu(mnist_train_cpu .÷ 2^bits)
@time pc = hclt(trunc_train, latents; num_cats = 256, pseudocount = 0.1, input_type = Categorical)
init_parameters(pc; perturbation = 0.4)
println("Number of free parameters: $(num_parameters(pc))")

print("Moving pc to GPU... ")
CUDA.@time bpc = CuBitsProbCircuit(pc);

num_epochs1       = 100
num_epochs2       = 250
num_epochs3       = 1
batch_size        = 512
pseudocount       = 0.1
param_inertia1    = 0.1 
param_inertia2    = 0.9
param_inertia3    = 0.95

@time begin
    mini_batch_em(bpc, mnist_train_gpu, num_epochs1; batch_size, pseudocount, 
                  param_inertia = param_inertia1, param_inertia_end = param_inertia2);
    mini_batch_em(bpc, mnist_train_gpu, num_epochs2; batch_size, pseudocount, 
                  param_inertia = param_inertia2, param_inertia_end = param_inertia3);
    full_batch_em(bpc, mnist_train_gpu, num_epochs3; batch_size, pseudocount);
end;

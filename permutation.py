import torch
import numpy as np
import matplotlib.pyplot as plt


batch_size = 128
input_dim = 128
output_dim = input_dim 
num_blocks = 4
num_experiments = 100

input_dim = input_dim - input_dim % num_blocks
output_dim = output_dim - output_dim % num_blocks


def create_block_diagonal_mat(dim1, dim2, num_blocks):
    mat = torch.zeros(dim1, dim2)
    block_dim1, block_dim2 = int(np.ceil(dim1 / num_blocks)), int(np.ceil(dim2 / num_blocks))
    for i in range(num_blocks):
        mat[i * block_dim1:(i + 1) * block_dim1, i * block_dim2:(i + 1) * block_dim2] = torch.randn([block_dim1, block_dim2])
    return mat

def get_blocks(mat, num_blocks):
    block_dim1, block_dim2 = int(np.ceil(mat.shape[0] / num_blocks)), int(np.ceil(mat.shape[1] / num_blocks))
    blocks = torch.zeros(num_blocks, block_dim1, block_dim2)
    for i in range(num_blocks):
        begin1, begin2 = i * block_dim1, i * block_dim2
        end1, end2 = min((i + 1) * block_dim1, mat.shape[0]), min((i + 1) * block_dim2, mat.shape[1])
        blocks[i, :end1 - begin1, :end2 - begin2] = mat[begin1:end1, begin2:end2]
    return blocks


standard_time = 0
block_time = 0



batch_size_list = [64, 128, 256, 512, 1024, 2048]
input_dim_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# for batch_size in batch_size_list:
#     speedups = []
#     for input_dim in input_dim_list:
#         output_dim = input_dim
#         for i in range(num_experiments):
#             x = torch.randn(batch_size, input_dim)
#             weight = create_block_diagonal_mat(input_dim, output_dim, num_blocks)
#             weight_blocks = get_blocks(weight, num_blocks)
#             perm1 = torch.randperm(input_dim)

#             standard_start = torch.cuda.Event(enable_timing=True)
#             standard_end = torch.cuda.Event(enable_timing=True)
#             standard_start.record()
#             out1 = x @ weight
#             standard_end.record()
#             torch.cuda.synchronize()
#             standard_time += standard_start.elapsed_time(standard_end)

#             block_start = torch.cuda.Event(enable_timing=True)
#             block_end = torch.cuda.Event(enable_timing=True)
#             block_start.record()
#             out2 = (x[:, perm1].view(num_blocks, batch_size, -1) @ weight_blocks).view(batch_size, -1)
#             block_end.record()
#             torch.cuda.synchronize()
#             block_time += block_start.elapsed_time(block_end)
            

#         print("Batch size: {}, Matrix Size: {}, Standard time: {}, Block time: {}, Speedup: {}".format(batch_size, input_dim, standard_time, block_time, standard_time / block_time))
#         speedups.append(standard_time / block_time)

#     plt.semilogx(input_dim_list, speedups)


with open("scripts/slurm-327169.out") as file:
    i = 0
    line = file.readline()
    while line:
        speedups = []
        if line.startswith("Batch size:"):
            speedups.append(float(line.split(" ")[-1][:-2]))
            for j in range(7):
                line = file.readline()
                speedups.append(float(line.split(" ")[-1][:-2]))
            if i > 0:
                plt.semilogx(input_dim_list, speedups)
            i += 1
            print(line)

        line = file.readline()

plt.xlabel("Matrix Dimension")
plt.ylabel("Speedup")
plt.title("Speedup of Block Diagonal Matrix Multiplication")
plt.savefig("figures/permutations/input_dim.pdf")
plt.legend(["Batch size: {}".format(batch_size) for batch_size in batch_size_list])
# plt.plot(batch_size_list, speedups)
plt.show()
import torch
import matplotlib.pyplot as plt


def simple_randomized_torch_svd(B, k=40):
    m, n = B.size()
    transpose = False
    if m < n:
        transpose = True
        B = B.transpose(0, 1).to(B.device)
        m, n = B.size()
    rand_matrix = torch.rand((n, k)).to(B.device)  # short side by k
    Q, _ = torch.linalg.qr(B @ rand_matrix)  # long side by k
    smaller_matrix = (Q.transpose(0, 1) @ B)  # k by short side
    U_hat, s, V = torch.svd(smaller_matrix, False)
    U = (Q @ U_hat)

    if transpose:
        return V.transpose(0, 1), s, U.transpose(0, 1)
    else:
        return U, s, V


def svd_experiment():
    mat_dim = 400
    rank = int(0.1 * mat_dim)
    num_experiments = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matrices = [torch.randn(mat_dim, rank).to(device) for i in range(num_experiments)]
    for i in range(num_experiments):
        matrices[i] = matrices[i] @ matrices[i].t()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_eigh = 0
    time_svd = 0
    time_randomized_svd = 0

    start.record()
    for mat in matrices:
        L, Q = torch.linalg.eigh(mat)
    end.record()
    torch.cuda.synchronize()
    time_eigh += start.elapsed_time(end)
    print("Error", torch.norm(matrices[-1] - Q @ torch.diag(L) @ Q.t()) / torch.norm(matrices[-1]))

    start.record()
    for mat in matrices:
        U, s, V = torch.linalg.svd(mat)
    end.record()
    torch.cuda.synchronize()
    time_svd += start.elapsed_time(end)
    print("Error", torch.norm(matrices[-1] - U @ torch.diag(s) @ V) / torch.norm(matrices[-1]))

    start.record()
    for mat in matrices:
        U, s, V = simple_randomized_torch_svd(mat)
    end.record()
    torch.cuda.synchronize()
    time_randomized_svd += start.elapsed_time(end)
    print("Error", torch.norm(matrices[-1] - U @ torch.diag(s) @ V.t()[:rank, :]) / torch.norm(matrices[-1]))

    print("eigh time: ", time_eigh / num_experiments)
    print("svd time: ", time_svd / num_experiments)
    print("randomized svd time: ", time_randomized_svd / num_experiments)


def sparsity_ratio_experiment():
    num_experiments = 10
    dense_start_timer = torch.cuda.Event(enable_timing=True)
    dense_end_timer = torch.cuda.Event(enable_timing=True)
    sparse_start_timer = torch.cuda.Event(enable_timing=True)
    sparse_end_timer = torch.cuda.Event(enable_timing=True)
    density_ratios = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    plt.figure()
    for dim in [128, 256, 512, 1024, 2048, 4096]:
        lef_sparse_speedups = []
        right_sparse_speedups = []
        original_sparse_matrices = torch.rand(num_experiments, dim, dim)
        dense_matrices = torch.randn(num_experiments, dim, dim)
        for density_ratio in density_ratios:
            csr_sparse_matrices = []
            sparse_matrices = []
            for i in range(num_experiments):
                mask = (original_sparse_matrices[i, :, :] < (density_ratio / 100)).to(torch.float)
                sparse_mat = (original_sparse_matrices[i, :, :] * mask)
                csr_sparse_matrices.append((sparse_mat).to_sparse_csr())
                sparse_matrices.append(sparse_mat)

            dense_start_timer.record()
            for i in range(num_experiments):
                dense_mat = dense_matrices[i, :, :]
                sparse_mat = sparse_matrices[i]
                result = sparse_mat @ dense_mat
            dense_end_timer.record()
            torch.cuda.synchronize()
            dense_time = dense_start_timer.elapsed_time(dense_end_timer)

            sparse_start_timer.record()
            for i in range(num_experiments):
                dense_mat = dense_matrices[i, :, :]
                sparse_mat = csr_sparse_matrices[i]
                result = sparse_mat @ dense_mat
            sparse_end_timer.record()
            torch.cuda.synchronize()
            left_sparse_time = sparse_start_timer.elapsed_time(sparse_end_timer)

            sparse_start_timer.record()
            for i in range(num_experiments):
                dense_mat = dense_matrices[i, :, :]
                sparse_mat = csr_sparse_matrices[i]
                result = dense_mat @ sparse_mat
            sparse_end_timer.record()
            torch.cuda.synchronize()
            right_sparse_time = sparse_start_timer.elapsed_time(sparse_end_timer)

            lef_sparse_speedups.append(dense_time / left_sparse_time)
            right_sparse_speedups.append(dense_time / right_sparse_time)
        print("Dim: ", dim)
        print("Left sparse speedups: ", lef_sparse_speedups)
        print("Right sparse speedups: ", right_sparse_speedups)
        plt.plot(density_ratios, lef_sparse_speedups, label="Left Sparse - dim:{}".format(dim))
        plt.plot(density_ratios, right_sparse_speedups, label="Right Sparse - dim:{}".format(dim))
    plt.legend()
    plt.xlabel("Density Ratio")
    plt.ylabel("Speedup")
    plt.show()


sparsity_ratio_experiment()
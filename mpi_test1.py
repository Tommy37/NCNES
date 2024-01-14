from mpi4py import MPI

# 初始化MPI环境

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 定义每个线程要执行的任务
def worker(rank):
    # 执行任务
    result = rank * 2
    
    # 将结果发送给其他线程
    for i in range(size):
        if i != rank:
            comm.send(result, dest=i)
    
    # 接收其他线程发送的结果
    for i in range(size):
        if i != rank:
            received_result = comm.recv(source=i)
            print(f"Thread {rank} received result {received_result} from thread {i}")

# 主线程调用worker函数
if rank < size:
    worker(rank)
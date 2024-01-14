from mpi4py import MPI

def distribute_work(n, num_processors):
    work_per_processor = n // num_processors
    remainder = n % num_processors
    
    # 计算每个处理器分配的任务数量
    work_counts = [work_per_processor] * num_processors
    for i in range(remainder):
        work_counts[i] += 1


    return work_counts

def find_thread(distribution, target_distribution):
    thread = -1
    for i, dist in enumerate(distribution):
        if dist == target_distribution:
            thread = i
            break
    return thread

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 假设有n个分布需要合理分配到4个处理器上
    n = 10

    if rank == 0:
        # 主进程分配任务给各个处理器
        #一个数组,代表主线程根据 distribute_work函数计算出的给各个线程的任务数量
        recv_counts = distribute_work(n, size)
    else:
        recv_counts = None
     
    # 各处理器接收任务数量, 比如当前线程4，接受了3个分布的计算任务，则返回3
    recv_count = comm.scatter(recv_counts, root=0)

    # 简单的假设10个分布是这样给出的
    distribution = [1,2,3,4,5,6,7,8,9,10] 

    # 假设每个线程的计算任务就是找到分布为2的分布在哪个线程
    result= [0] * recv_count
    for i in range(recv_count):
        thread = find_thread(distribution, 2)  # 查找分布为2的线程
        result[i]=thread
    # 各线程返回计算结果给主进程,在主线程打印结果
    print(comm.gather(result, root=0))
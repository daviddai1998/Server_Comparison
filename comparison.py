import random
from sklearn.utils import murmurhash3_32
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
# random.seed(1)


num_servers = 5
Pf = [0.1, 0.05]
Pa = [0.01, 0.005]


class SPOCA:

    def __init__(self, n, pf, pa, t):
        self.num_servers = n
        self.fail_rate = pf
        self.add_rate = pa
        self.total_time = 0
        self.capacity = []
        self.servers = {}
        self.status = [0]
        self.hash_size = int(10 * t)
        self.hash_space = [-1 for _ in range(self.hash_size + 1)]
        self.total_size = 0

    def add_server(self, server_id, size):
        hash_value = murmurhash3_32(server_id) % self.hash_size
        # not sure how to deal with cycle
        while True:
            flag = True
            for p in range(size):
                if self.hash_space[hash_value + p] != -1:
                    flag = False
            if flag:
                for q in range(size):
                    self.hash_space[hash_value + q] = server_id
                self.status.insert(server_id, 1)
                self.servers[server_id] = []
                break
            else:
                hash_value = murmurhash3_32(hash_value) % self.hash_size
        return

    def add_element(self, vid):
        start_spoca = time.time()
        h_val = murmurhash3_32(vid) % self.hash_size
        # not sure how to deal with cycle
        cnt = 0
        while True:
            if cnt > 50:
                cnt = 0
                h_val = (h_val + random.randint(1, 20)) % self.hash_size
            if self.hash_space[h_val] == -1:
                h_val = murmurhash3_32(h_val) % self.hash_size
                cnt += 1
            else:
                server_id = self.hash_space[h_val]
                self.servers[server_id].append(vid)
                break
        end_spoca = time.time()
        self.total_time += end_spoca - start_spoca
        return

    # server_id must be present in servers dictionary
    def fail_server(self, server_id):
        self.status[server_id] = 0
        for _ in range(self.hash_size):
            if self.hash_space[_] == server_id:
                self.hash_space[_] = -1
        for vid in self.servers[server_id]:
            self.add_element(vid)
            # print("zhelima")
        self.servers.pop(server_id)
        return


class BoundedConsistentHashing:
    def __init__(self, n, pf, pa, t):
        self.num_servers = n
        self.fail_rate = pf
        self.add_rate = pa
        self.total_time = 0
        self.capacity = []
        self.overhead = []
        self.servers = {}
        self.status = [0]
        self.hash_size = int(10 * t)
        self.hash_space = [-1 for _ in range(self.hash_size)]
        self.total_size = 0
        return

    def add_server(self, server_id, size):
        # print("add")
        h_val = murmurhash3_32(server_id) % self.hash_size
        cnt = 0
        # not sure how to deal with cycle
        while True:
            flag = True
            if cnt > 50:
                cnt = 0
                h_val = (h_val + random.randint(1, 30)) % self.hash_size
            for p in range(size):
                if self.hash_space[h_val + p] != -1:
                    flag = False
            if flag:
                for q in range(size):
                    self.hash_space[h_val + q] = server_id
                break
            else:
                h_val = murmurhash3_32(h_val) % self.hash_size
                cnt += 1
        self.status.insert(server_id, 1)
        self.overhead.append(-size)
        self.servers[server_id] = []
        return

    def add_element(self, vid):
        start_bound = time.time()
        h_val = murmurhash3_32(vid) % self.hash_size
        cnt = 0
        # not sure how to deal with cycle
        flag = True
        for _ in self.overhead:
            if _ < 0:
                flag = False
                break
        if flag:
            least = min(self.overhead)
            server_id = self.overhead.index(least)
            self.overhead[server_id] += 1
            self.servers[server_id + 1].append(vid)

        else:
            while True:
                if cnt > 50:
                    cnt = 0
                    h_val = (h_val + random.randint(1, 30)) % self.hash_size
                if self.hash_space[h_val] != -1:
                    h_val = murmurhash3_32(h_val) % self.hash_size
                    cnt += 1
                else:
                    ptr = (h_val - 1) % self.hash_size
                    while True:
                        # print(ptr, h_val)
                        if ptr == h_val:
                            print("There is no server in the hash space")
                            server_id = -1
                            break
                        elif self.hash_space[ptr] != -1:
                            server_id = self.hash_space[ptr]
                            break
                        else:
                            ptr = (ptr - 1) % self.hash_size
                    # server_id = self.search_server(h_val)
                    if server_id == -1:
                        print("no server in the hash space")
                        break
                    if self.overhead[server_id - 1] < 0:
                        self.overhead[server_id - 1] += 1
                        self.servers[server_id].append(vid)
                        break
                    else:
                        h_val = murmurhash3_32(h_val) % self.hash_size
                        cnt += 1
        end_bound = time.time()
        # print(end_bound - start_bound, self.status.count(1))
        self.total_time += end_bound - start_bound
        return

    def fail_server(self, server_id):
        # print("fail")
        self.status[server_id] = 0
        for p in range(self.hash_size):
            if self.hash_space[p] == server_id:
                self.hash_space[p] = -1
        for vid in self.servers[server_id]:
            start = time.time()
            self.add_element(vid)
            end = time.time()
            self.total_time += end - start
        self.servers.pop(server_id)
        self.overhead[server_id - 1] = float("inf")
        return

    # def search_server(self, h_val):
    #     ptr = (h_val - 1) % self.hash_size
    #     while True:
    #         # print(ptr, h_val)
    #         if ptr == h_val:
    #             # print("There is no server in the hash space")
    #             return -1
    #         elif self.hash_space[ptr] != -1:
    #             server_id = self.hash_space[ptr]
    #             break
    #         else:
    #             ptr = (ptr - 1) % self.hash_size
    #     return server_id


class MultipleConsistentHashing:
    def __init__(self, n, pf, pa, t):
        self.num_servers = n
        self.fail_rate = pf
        self.add_rate = pa
        self.total_time = 0
        self.capacity = []
        self.servers = {}
        self.status = [0]
        self.hash_size = int(10 * t)
        self.hash_space = [-1 for _ in range(self.hash_size)]
        self.total_size = 0
        return

    def add_server(self, server_id, size):
        for p in range(size):
            h_val = murmurhash3_32(server_id, p) % self.hash_size
            cnt = 0
            # not sure how to deal with cycle
            while True:
                if cnt > 50:
                    cnt = 0
                    h_val = (h_val + random.randint(1, 30)) % self.hash_size
                if self.hash_space[h_val] != -1:
                    h_val = murmurhash3_32(h_val, p) % self.hash_size
                    cnt += 1
                else:
                    self.hash_space[h_val] = server_id
                    break
        self.status.insert(server_id, 1)
        self.servers[server_id] = []
        return

    def add_element(self, vid):
        start_mult = time.time()
        h_val = murmurhash3_32(vid) % self.hash_size
        cnt = 0
        # not sure how to deal with cycle
        while True:
            # print("caonimmmmmmmma")
            if cnt > 80:
                cnt = 0
                h_val = (h_val + random.randint(1, 40)) % self.hash_size
            if self.hash_space[h_val] != -1:
                h_val = murmurhash3_32(h_val) % self.hash_size
                cnt += 1
            else:
                # print("here?")
                server_id = self.search_server(h_val)
                # print(server_id)
                # print(vid)
                # print(self.servers.keys())
                self.servers[server_id].append(vid)
                break
        # print("zhelimmmmma")
        end_mult = time.time()
        self.total_time += end_mult - start_mult
        return

    def fail_server(self, server_id):
        # print("111111")
        self.status[server_id] = 0
        for p in range(self.hash_size):
            if self.hash_space[p] == server_id:
                self.hash_space[p] = -1
        for vid in self.servers[server_id]:
            self.add_element(vid)
            # print("end")
        self.servers.pop(server_id)
        # print("2222222222")
        return

    # we store server as string and tasks as int so we can distinguish them
    def search_server(self, h_val):
        ptr = (h_val - 1) % self.hash_size
        while True:
            # print(ptr)
            if ptr == h_val:
                print("There is no server in the hash space")
                return -1
            elif self.hash_space[ptr] != -1:
                server_id = self.hash_space[ptr]
                break
            else:
                ptr = (ptr - 1) % self.hash_size
        return server_id


def run(pa, pf):
    spoca_frac_0_03 = [0 for k in range(1000)]
    spoca_frac_03_07 = [0 for k in range(1000)]
    spoca_frac_07_1 = [0 for k in range(1000)]
    spoca_frac_1_13 = [0 for k in range(1000)]
    spoca_frac_13 = [0 for k in range(1000)]

    bounded_frac_0_03 = [0 for k in range(1000)]
    bounded_frac_03_07 = [0 for k in range(1000)]
    bounded_frac_07_1 = [0 for k in range(1000)]
    bounded_frac_1_13 = [0 for k in range(1000)]
    bounded_frac_13 = [0 for k in range(1000)]

    multiple_frac_0_03 = [0 for k in range(1000)]
    multiple_frac_03_07 = [0 for k in range(1000)]
    multiple_frac_07_1 = [0 for k in range(1000)]
    multiple_frac_1_13 = [0 for k in range(1000)]
    multiple_frac_13 = [0 for k in range(1000)]

    spocatime = []
    boundtime = []
    copytime = []

    spoca1 = SPOCA(5, pa, pf, 1000)
    bounded1 = BoundedConsistentHashing(5, pa, pf, 1000)
    multiple1 = MultipleConsistentHashing(5, pa, pf, 1000)

    for i in range(1, 6):
        server_size = random.randint(1, 3)
        spoca1.capacity.append(server_size)
        spoca1.total_size += server_size
        spoca1.add_server(i, server_size)
        bounded1.capacity.append(server_size)
        bounded1.total_size += server_size
        bounded1.add_server(i, server_size)
        multiple1.capacity.append(server_size)
        multiple1.total_size += server_size
        multiple1.add_server(i, server_size)

    for j in range(1000):
        # add and fail server
        # spoca1.total_time = 0
        # bounded1.total_time = 0
        # multiple1.total_time = 0
        rate = random.random()
        if rate <= pa:
            server_size = random.randint(1, 3)
            spoca1.num_servers += 1
            spoca1.add_server(spoca1.num_servers, server_size)
            spoca1.capacity.append(server_size)
            bounded1.num_servers += 1
            bounded1.add_server(bounded1.num_servers, server_size)
            bounded1.capacity.append(server_size)
            multiple1.num_servers += 1
            multiple1.add_server(multiple1.num_servers, server_size)
            multiple1.capacity.append(server_size)
        rate2 = random.random()
        if rate2 <= pf:
            spoca_id = random.randint(1, spoca1.num_servers)
            # print(spoca1.status)
            # possible deadlock, keep track of the id of active servers
            while spoca1.status[spoca_id] == 0:
                # print("cnm1")
                spoca_id = random.randint(1, spoca1.num_servers)
            spoca1.fail_server(spoca_id)

            bounded_id = random.randint(1, bounded1.num_servers)
            # possible deadlock, keep track of the id of active servers
            while bounded1.status[bounded_id] == 0:
                # print("cnm1")
                bounded_id = random.randint(1, bounded1.num_servers)
            bounded1.fail_server(bounded_id)

            multiple_id = random.randint(1, multiple1.num_servers)
            # print(spoca1.status)
            # possible deadlock, keep track of the id of active servers
            while multiple1.status[multiple_id] == 0:
                # print("cnm1")
                multiple_id = random.randint(1, multiple1.num_servers)
            multiple1.fail_server(multiple_id)
            # print("zhala")

        # spoca_start = time.time()
        # spoca1.add_element(j)
        # spoca_end = time.time()
        # spoca1.total_time += spoca_end - spoca_start
        # bound_start = time.time()
        # bounded1.add_element(j)
        # bound_end = time.time()
        # bounded1.total_time += bound_end - bound_start
        # copy_start = time.time()
        # multiple1.add_element(j)
        # copy_end = time.time()
        # multiple1.total_time += copy_end - copy_start

        spoca1.add_element(j)
        bounded1.add_element(j)
        multiple1.add_element(j)

        for s in range(1, len(spoca1.status)):
            if spoca1.status[s] == 1:
                frac = len(spoca1.servers[s]) / float(spoca1.capacity[s - 1])
                if 0 <= frac < 0.3:
                    spoca_frac_0_03[j] += 1
                elif 0.3 <= frac < 0.7:
                    spoca_frac_03_07[j] += 1
                elif 0.7 <= frac < 1.0:
                    spoca_frac_07_1[j] += 1
                elif 1.0 <= frac < 1.3:
                    spoca_frac_1_13[j] += 1
                else:
                    spoca_frac_13[j] += 1
        spoca_frac_0_03[j] /= spoca1.status.count(1)
        spoca_frac_03_07[j] /= spoca1.status.count(1)
        spoca_frac_07_1[j] /= spoca1.status.count(1)
        spoca_frac_1_13[j] /= spoca1.status.count(1)
        spoca_frac_13[j] /= spoca1.status.count(1)

        for s in range(1, len(bounded1.status)):
            if bounded1.status[s] == 1:
                # print(s)
                # print(bounded1.servers.keys())
                # print(len(bounded1.status))
                frac = len(bounded1.servers[s]) / float(bounded1.capacity[s - 1])
                if 0 <= frac < 0.3:
                    bounded_frac_0_03[j] += 1
                elif 0.3 <= frac < 0.7:
                    bounded_frac_03_07[j] += 1
                elif 0.7 <= frac < 1.0:
                    bounded_frac_07_1[j] += 1
                elif 1.0 <= frac < 1.3:
                    bounded_frac_1_13[j] += 1
                else:
                    bounded_frac_13[j] += 1
        bounded_frac_0_03[j] /= bounded1.status.count(1)
        bounded_frac_03_07[j] /= bounded1.status.count(1)
        bounded_frac_07_1[j] /= bounded1.status.count(1)
        bounded_frac_1_13[j] /= bounded1.status.count(1)
        bounded_frac_13[j] /= bounded1.status.count(1)

        for s in range(1, len(multiple1.status)):
            if multiple1.status[s] == 1:
                # print(s)
                # print(multiple1.servers.keys())
                # print(len(multiple1.status))
                frac = len(multiple1.servers[s]) / float(multiple1.capacity[s - 1])
                if 0 <= frac < 0.3:
                    multiple_frac_0_03[j] += 1
                elif 0.3 <= frac < 0.7:
                    multiple_frac_03_07[j] += 1
                elif 0.7 <= frac < 1.0:
                    multiple_frac_07_1[j] += 1
                elif 1.0 <= frac < 1.3:
                    multiple_frac_1_13[j] += 1
                else:
                    multiple_frac_13[j] += 1
        multiple_frac_0_03[j] /= multiple1.status.count(1)
        multiple_frac_03_07[j] /= multiple1.status.count(1)
        multiple_frac_07_1[j] /= multiple1.status.count(1)
        multiple_frac_1_13[j] /= multiple1.status.count(1)
        multiple_frac_13[j] /= multiple1.status.count(1)
        spocatime.append(spoca1.total_time)
        boundtime.append(bounded1.total_time)
        copytime.append(multiple1.total_time)
        # print(spoca1.status.count(1), len(spoca1.status) - 1)
        # print(bounded1.status.count(1), len(bounded1.status) - 1)
        # print(multiple1.status.count(1), len(multiple1.status) - 1)

    print(spoca1.total_time)
    print(bounded1.total_time)
    print(multiple1.total_time)

    # f = ["0-0.3", "0.3-0.7", "1-1.3", "1.3+"]
    # spocaline = [spoca_frac_0_03, spoca_frac_03_07, spoca_frac_1_13, spoca_frac_13]
    # boundline = [bounded_frac_0_03, bounded_frac_03_07, bounded_frac_1_13, bounded_frac_13]
    # copyline = [multiple_frac_0_03, multiple_frac_03_07, multiple_frac_1_13, multiple_frac_13]
    # for itr in range(4):
    #     df = pd.DataFrame({'spoca': np.array(spocaline[itr]),
    #                        'bounded': np.array(boundline[itr]),
    #                        'copy': np.array(copyline[itr])})
    #     plt.title("PA = " + str(pa) + ", PF = " + str(pf) + ", frequency = " + str(f[itr]))
    #     plt.xlabel("time")
    #     plt.ylabel("fraction of servers")
    #     plt.plot('spoca', data=df, color='red', linewidth=1)
    #     plt.plot('bounded', data=df, color='blue', linewidth=1)
    #     plt.plot('copy', data=df, color='green', linewidth=1)
    #     plt.legend()
    #     plt.show()

    df_time = pd.DataFrame({
        "spoca": np.array(spocatime),
        "bounded": np.array(boundtime),
        "copy": np.array(copytime)
    })
    plt.title("PA = " + str(pa) + ", PF = " + str(pf))
    plt.xlabel("time")
    plt.ylabel("overhead")
    plt.plot('spoca', data=df_time, color='red', linewidth=1)
    plt.plot('bounded', data=df_time, color='blue', linewidth=1)
    plt.plot('copy', data=df_time, color='green', linewidth=1)
    plt.legend()
    plt.show()


run(0.1, 0.01)
# run(0.05, 0.01)
# run(0.1, 0.005)
# run(0.05, 0.005)

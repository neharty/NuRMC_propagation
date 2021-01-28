from multiprocessing import Process, Queue
import time

class A:

    def __init__(self, init):
        self.a = init

    def func(self, b):
        self.a = self.a + b
    
    def par_func(self, q, b):
        print('starting')
        self.func(b)
        q.put(self.a)
        time.sleep(10)
        print('whew! all done')

foo = A(2)
bar = A(6)
print('initial:', foo.a, bar.a)

q1, q2 = Queue(), Queue()
p1, p2 = Process(target=foo.par_func, args=(q1, 3)), Process(target=bar.par_func, args=(q2, 4))
p1.start()
p2.start()
p1.join()
p2.join()

foo.a = q1.get()
bar.a = q2.get()

print('final:', foo.a, bar.a)


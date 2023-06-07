from queue import Queue
from threading import Thread, Event
import time

class Worker(Thread):
    def __init__(self, taskQ, threadName):
        Thread.__init__(self, name=threadName)
        self.taskQ = taskQ
        self._stop_event = Event()
        self.start()
    
    def run(self):
        threadName = self.getName()
        threadId   = self.ident
        print(f"Thread spawned name:{threadName} id:{threadId}")
        while not self.stopped():
            try:
                func = self.taskQ.get(timeout = 5)
                if func != None:
                    func(threadName, threadId)
                    self.taskQ.task_done()
                print("trying")
            except Exception as e:
                print(e)
            finally:
                continue
        print("Thread stopped!!")
    
    def stop(self):
        self._stop_event.set()
    
    def stopped(self):
        return self._stop_event.is_set()

class ThreadPool:
    def __init__(self, numThreads):
        self.taskQ = Queue(numThreads)
        self.workers = []
        self.numThreads = numThreads
        #intialize threads and the
        for i in range(self.numThreads):
            worker = Worker(self.taskQ, f"worker_{i}")
            self.workers.append(worker)
    
    def add_task(self, func):
        self.taskQ.put(func)
    
    def waitForAllTasksCompleted(self):
        self.taskQ.join()
        #while not self.taskQ.empty():
            #sleep for 50 ms
            #time.sleep(50.0/1000.0)

    def shutDownPool(self):
        self.waitForAllTasksCompleted()
        for worker in self.workers:
            worker.stop()

        for worker in self.workers:
            worker.join()



class Test:
    def __init__(self, msg):
        self.msg = msg
    
    def printMsg(self):
        time.sleep(2)
        print(self.msg)

if __name__ == "__main__":
    pool = ThreadPool(3)
    t1 = Test("hi")
    t2 = Test("hello world")
    t3 = Test("good")
    l1 = lambda name, id : t1.printMsg()
    l2 = lambda name, id : t2.printMsg()
    l3 = lambda name, id : t3.printMsg()

    pool.add_task(l1)
    pool.add_task(l2)
    pool.add_task(l3)
    pool.waitForAllTasksCompleted()
    pool.shutDownPool()
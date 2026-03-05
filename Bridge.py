import queue, copy
from PyQt5.QtCore import (QObject, QMutex)

class Mutex(QMutex):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        super().lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().unlock()

class Subscriber(QObject):
    def __init__(self, topic_name:str, maxsize=1) -> None:
        self.message_list = queue.Queue(maxsize=maxsize)
        self.mutex = Mutex()
        self.name = topic_name
        

        if topic_name not in subsicribe_list.keys():
            subsicribe_list.add_topic(topic_name)
        subsicribe_list.append(self.name, self)

    def get_message(self, timeout=None):
        return self.message_list.get(timeout=timeout)
    
    def pub_message(self, message, timeout=None):
        if self.message_list.full():
            self.message_list.get()  
        self.message_list.put(message, timeout=timeout)
    
    def is_empty(self):
        return self.message_list.empty()


class Subsicribe_list(QObject):
    def __init__(self):
        self.sub_list = {}  
        self.mutex = Mutex()

    def append(self, name:str,  subscriber:Subscriber):
        with self.mutex:
            self.sub_list[name].append(subscriber)
    
    def add_topic(self, name:str):
        with self.mutex:
            self.sub_list[name] = []

    def keys(self):
        with self.mutex:
            keys = self.sub_list.keys()
        return keys
    
    def get_subscribers(self, name:str):
        with self.mutex:
            sub_list = self.sub_list[name].copy()
        return sub_list


subsicribe_list = Subsicribe_list()


class Publisher(QObject):
    def __init__(self, name:str):
        self.name = name
        self.mutex = QMutex()

      
        if name not in subsicribe_list.keys():
            subsicribe_list.add_topic(name)

    def publish(self, message, timeout:float | None=None):
        sub_list = subsicribe_list.get_subscribers(self.name)
        for subsicriber in sub_list:
            now_piece = copy.deepcopy(message)  # 《引以为戒》
            subsicriber.pub_message(now_piece, timeout=timeout)   

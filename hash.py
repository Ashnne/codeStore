import hashlib
import os
import pickle

class HashDatabase:
    def __init__(self, filename):
        self.filename = filename
        self.hash_set = set()
        # 初始化时加载现有哈希值
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                self.hash_set = pickle.load(f)
        else:
            # 创建空文件
            open(self.filename, 'wb').close()

    def check(self,input_str):
        if input_str in self.hash_set:
            return True
        else:
            return False

    def add(self, input_str):
        if input_str in self.hash_set:
            return False  # 已存在，返回False

        # 更新内存中的哈希集合
        self.hash_set.add(input_str)
        return True  # 新增成功，返回True
    
    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.hash_set,f)

# 测试用例
if __name__ == "__main__":
    db = HashDatabase("hashes.db")
    
    print(db.add("Hello World"))    # True（首次添加）
    print(db.add("Hello World"))    # False（重复）
    print(db.add("Another string")) # True
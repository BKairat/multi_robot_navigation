import numpy as np

class History:
    def __init__(self, first: dict,lenght: int = 4):
        self.lenght = lenght
        self.data = {
            #  lasers
            "ol": [first["ol"] for _ in range(lenght)],
            #  relative orientation to the goal as dir vector
            "og": [first["og"] for _ in range(lenght)],
            #  linear and angular velocity 
            "ov": [first["ov"] for _ in range(lenght)],
            #  euclidian dstance to goal.
            "od": [first["od"] for _ in range(lenght)],
            "d_0": first["od"]
        }
        
    def add(self, observation: dict):        
        self.data["ol"].pop(0)
        self.data["og"].pop(0)
        self.data["ov"].pop(0)
        self.data["od"].pop(0)
        
        self.data["ol"].append(observation["ol"])
        self.data["og"].append(observation["og"])
        self.data["ov"].append(observation["ov"])
        self.data["od"].append(observation["od"])
    
    def item(self, i: str):
        return self.data[i]
    
    def get(self):
        return {
            #  lasers
            "ol": np.concatenate(self.data["ol"]),
            #  relative orientation to the goal as dir vector
            "og": np.concatenate(self.data["og"]),
            #  linear and angular velocity 
            "ov": np.concatenate(self.data["ov"]),
            #  euclidian dstance to goal.
            "od": np.concatenate(self.data["od"]),
            
            "d_0": self.data["d_0"]
        }
        
    def __len__(self):
        return self.lenght
    
    def get_vectors(self):
        return (np.reshape(np.concatenate(self.data["ol"]), (1, 64)),
                np.concatenate(self.data["od"]),
                np.concatenate(self.data["og"]),
                np.concatenate(self.data["ov"]))
        
    
    def __str__(self):
        data = self.get()
        return f"====\nod: {data['od']}"
    # ol: {data['ol'][:3]}...{data['ol'][-3:]} \nog: {data['og']}\nov: {data['ov']}\n
    
class MyHistory:
    def __init__(self, first: dict, lenght: int = 4):
        self.lenght = lenght
        self.data = {
            #  lasers
            "ol": [first["ol"] for _ in range(lenght)],
            #  robot possition and orientation relative to goal
            "op": [first["op"]],
        }
        
    def add(self, observation: dict):        
        self.data["ol"].pop(0)
        
        self.data["ol"].append(observation["ol"])
        self.data["op"].append(observation["op"])
    
    def item(self, i: str):
        return self.data[i]
    
    def get(self):
        return {
            #  lasers
            "ol": np.array(self.data["ol"][-1]),
            #  position and orientation
            "op": self.data["op"]
        }
        
    def __len__(self):
        return self.lenght
    
    def get_vectors(self):
        # print(self.data["op"])
        return (np.array(self.data["ol"]),
                np.array(self.data["op"][-1]))
        
    def __str__(self):
        data = self.get()
        return f"====\nol: {data['ol']}\nod: {data['od']}"
    
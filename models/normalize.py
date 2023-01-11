
class Normalize():
    def __init__(self, mean=None, std=None ):
        self.mean = mean
        self.std = std
    def transform_data(self, data):
        return (data-self.mean) / self.std

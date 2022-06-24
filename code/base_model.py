class BaseModel:
    def __init__(self, name, features):
        self.name = name
        self.continuous_features = []
        self.cat_features = []
        self.all_features = []
        for f in features:
            if f.categorical:
                self.cat_features.append(f.name)
            else:
                self.continuous_features.append(f.name)
            self.all_features.append(f.name)

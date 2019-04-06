import json


class RunningAverage:
    def __init__(self):
        self.count = 0
        self.total = 0

    def update(self, num):
        self.count += 1
        self.total += num

    def reset(self):
        self.count = 0
        self.total = 0

    def __call__(self, *args, **kwargs):
        try:
            avg = self.total / self.count
        except ZeroDivisionError:
            avg = 'NaN'
        return avg


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def running_average_test():
    ra = RunningAverage()
    print(ra())


if __name__ == '__main__':
    running_average_test()

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


def running_average_test():
    ra = RunningAverage()
    print(ra())


if __name__ == '__main__':
    running_average_test()

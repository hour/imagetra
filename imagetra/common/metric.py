from dataclasses import dataclass
from datetime import timedelta
import time

@dataclass
class Timer:
    time: float=None
    tracks={}

    def format(self):
        totaltime=sum(self.tracks.values())
        output = f'Average time: {str(timedelta(seconds=totaltime/len(self.tracks)))}\n'
        output += f'Total time: {str(timedelta(seconds=totaltime))}'
        return output

    def start(self):
        self.time = time.time()
        return self

    def track(self, name: str=None):
        assert(self.time is not None), 'Timer is not started yet'
        new_time = time.time()
        name = name if name is not None else len(self.tracks)
        self.tracks[name] = new_time - self.time
        self.time = new_time

    def detail(self):
        output = 'name\tduration\n'
        for name, duration in self.tracks.items():
            output += f'{name}\t{duration}\n'
        return output

# import random
# timer = Timer()
# timer.start()
# for i in range(5):
#     time.sleep(random.randint(1,2))
#     timer.track()
#     print(timer.format())

# print(timer.detail())
import logging, sys

def get_logger(name):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)

class LogFile:
    def __init__(self, path=None) -> None:
        self.logfile = None if path is None else open(path, 'w') 

    def print(self, msg):
        print(msg, file=self.logfile)
        if self.logfile is not None:
            self.logfile.flush()
    
    def close(self):
        if self.logfile is not None:
            self.logfile.close()
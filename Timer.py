import  time
import sys
import calendar
from datetime import datetime, timedelta


class MTimer:
    def __init__(self, message=''):
        if sys.platform == 'win32':
            self.t1 = time.clock()
        else:
            self.t1 = time.time()

        self.message = message


    def __enter__(self):
        return self


    def setmessage(self, message):
        self.message = message

    def end(self):
        if sys.platform == 'win32':
            t = time.clock() - self.t1
        else:
            t = time.time() - self.t1
        if t > 0.1:
            print(self.message, '{0:.2f}'.format(t), "s")
        elif t > 0.001:
            print(self.message, '{0:.2f}'.format(t * 1000), "ms")
        else :
            print(self.message, '{0:.2f}'.format(t * 1000000), "mcs")


    def __exit__(self, exc_type, exc_value, traceback):
        self.end()
        pass

# MTimer

def utc_to_local(utc_dt):
    # get integer timestamp to avoid precision lost
    timestamp = calendar.timegm(utc_dt.timetuple())
    local_dt = datetime.fromtimestamp(timestamp)
    assert utc_dt.resolution >= timedelta(microseconds=1)
    return local_dt.replace(microsecond=utc_dt.microsecond)


def intdate2str(d):
    return  "%d-%02d-%02d" % (d / 10000, (d / 100) % 100, d % 100)

def unixtime2str(t, wantMSec=True):
    it = int(t)
    h = it / 3600
    m = (it / 60) % 60
    s = it % 60
    ms = int((t - it) * 1000.0)
    res = "%02d:%02d:%02d" % (h, m, s)
    if wantMSec: res += ".%03d" % ms
    return res


def utc_to_local(utc_dt):
    # get integer timestamp to avoid precision lost
    timestamp = calendar.timegm(utc_dt.timetuple())
    local_dt = datetime.fromtimestamp(timestamp)
    assert utc_dt.resolution >= timedelta(microseconds=1)
    return local_dt.replace(microsecond=utc_dt.microsecond)


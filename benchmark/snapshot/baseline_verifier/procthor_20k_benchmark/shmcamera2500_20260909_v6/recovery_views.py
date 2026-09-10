"""Bounded RGB reacquisition for external path policies, never GT obstacle repair."""
import math
from exact_executor import wrap


class RecoveryViews:
    def __init__(self, maximum=6):
        self.maximum=maximum
        self.reset()

    def reset(self):
        self.attempts=0
        self.base=None

    def next_turn(self, yaw, target):
        if self.attempts>=self.maximum: return None
        if self.base is None: self.base=yaw
        offsets=(30.,-30.,60.,-60.,90.,-90.)
        bearing=math.atan2(float(target[0]),float(target[1]))
        if self.attempts==0 and abs(bearing)>math.radians(35):
            delta=bearing
        else:
            delta=wrap(self.base+math.radians(offsets[self.attempts])-yaw)
        self.attempts+=1
        if abs(delta)<1e-6:
            return self.next_turn(yaw,target)
        return max(-math.pi/3,min(math.pi/3,delta))

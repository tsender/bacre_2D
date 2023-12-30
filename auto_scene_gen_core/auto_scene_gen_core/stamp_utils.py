import math
import builtin_interfaces.msg as builtin_interfaces

def do_ros_stamps_match(stamp1: builtin_interfaces.Time, stamp2: builtin_interfaces.Time):
    return stamp1.sec == stamp2.sec and stamp1.nanosec == stamp2.nanosec

def get_stamp_dt(stamp1: builtin_interfaces.Time, stamp2: builtin_interfaces.Time):
    """Return the floating decimal time delta from stamp1 to stamp2. That is, stamp2 - stamp1."""
    sec_diff = stamp2.sec - stamp1.sec
    nsec_diff = stamp2.nanosec - stamp1.nanosec
    return float(sec_diff) + float(nsec_diff / 1e9)

def add_time_to_stamp(stamp: builtin_interfaces.Time, add_time: float):
    """Return a new builtin_interfaces/Time message with add_time added to stamp
    
    Args:
        - stamp: Time stamp to add time to
        - add_time: Amount of time, as decimal number, to add to stamp
    """
    res = int(1e9)
    y,Y = math.modf(add_time) # Want: Y + y/res, with y,Y,res being ints
    Y = round(Y)
    y = round(y * res) # Now have Y + y/res
    z = stamp.nanosec + y
    carry = 0
    if z >= res:
        z -= res
        carry = 1
    elif z < 0:
        z += res
        carry = -1

    new_stamp = builtin_interfaces.Time()
    new_stamp.sec = stamp.sec + Y + carry
    new_stamp.nanosec = z
    return new_stamp
    
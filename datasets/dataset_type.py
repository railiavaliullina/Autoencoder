from enum import Flag

class DatasetType(Flag):
    Train = 1 << 0
    Test = 1 << 1
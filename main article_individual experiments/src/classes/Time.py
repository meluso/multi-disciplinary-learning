# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:22:03 2022

@author: John Meluso
"""

# Import libraries
import datetime as dt


class Time(object):
    
    def __init__(self):
        self.start = dt.datetime.now()
        self.array = [self.start]
        self.connector = ' - '
        
    def begin(self, obj_type, script_activity, obj):
        self.obj_type = obj_type
        self.last = self.array[-1]
        self.now = dt.datetime.now()
        self.array.append(self.now)
        time = self.elapsed()
        message = f'Starting {script_activity} for {self.obj_type} {obj}.'
        print(time + self.connector + message)
        
    def update(self, event):
        self.last = self.array[-1]
        self.now = dt.datetime.now()
        self.array.append(self.now)
        time = self.elapsed()
        message = f'\t{event} at {time}'
        print(time + self.connector + message)
    
    def end(self, script_activity, obj):
        self.last = self.array[-1]
        self.now = dt.datetime.now()
        self.array.append(self.now)
        time = self.elapsed()
        message = f'{script_activity} complete for {self.obj_type} {obj}.'
        print(time + self.connector + message)
        
    def elapsed(self):
        return f'{self.now - self.start}'
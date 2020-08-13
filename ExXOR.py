# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:12:41 2020

@author: Mohammed Maaz Sibhai
"""


import numpy as np
from Perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,0]))
labels = np.array([0,1,1,0])
perceptron = Perceptron(2)
perceptron.train(training_inputs,labels)
inputs = np.array([1,0])
print(perceptron.predict(inputs))
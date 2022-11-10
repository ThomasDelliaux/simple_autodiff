# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 23:36:29 2022

@author: Thomas
"""
import numpy as np


class Node:
    
    def __init__(self, val, parents=None):
        if parents == None:
            parents =[]
        self.val = val
        self.parents_grad = parents
        self.grad = 0.0
    
    def backprop(self, grad_from_above):
        self.grad+=grad_from_above
        
        for parents,grad in self.parents_grad:
            parents.backprop(grad*grad_from_above)
        
    def backward(self):
        self.grad = 1
        self.backprop(self.grad)
    
    def __str__(self):
        return "Node( val =% f,grad =% f )"%(self.val,self.grad)
    
    def __add__(self,other_node):
        return Node(self.val+other_node.val,parents=[(self,1.0), (other_node,1.0)])
        
    def __mul__(self, other_node):
        return Node(self.val*other_node.val,parents=[(self,other_node.val), (other_node,self.val)])
    
    def __pow__(self, other_node):
        a = self.val        
        b = other_node.val
        return Node(a**b,parents=[(self,b*(a**(b-1))),(other_node,np.log(a)*(a**b))])
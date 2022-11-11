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
        self.grad = grad_from_above
        
        for parents,grad in self.parents_grad:
            parents.backprop(grad*grad_from_above)
        
    def backward(self):
        self.grad = 1
        self.backprop(self.grad)
    
    def __str__(self):
        return "Node( val =% f,grad =% f )"%(self.val,self.grad)
    
    def __add__(self,other_node):
        print(self)
        self,other_node = self.__check_type(other_node)
        return Node(self.val+other_node.val,parents=[(self,1.0), (other_node,1.0)])
    
    def __sub__(self,other_node):
        return Node(self.val-other_node.val,parents=[(self,1.0),(other_node,-1)])
        
    def __mul__(self, other_node):
        return Node(self.val*other_node.val,parents=[(self,other_node.val), (other_node,self.val)])
    
    def __pow__(self,power):
        a = self.val        
        return Node(a**power,parents=[(self,power*(a**(power-1)))])
    
    def __neg__(self):
        return Node(-self.val,parents=[(self,-1)])
    
    def __truediv__(self,other_node):
        a=self.val
        b=other_node.val
        return Node(a/b,parents=[(self,1/b),(other_node,-a/(b**2))])
    
    def __check_type(self,other_node):
        if type(self)!=Node and type(other_node)!=Node:
            self=Node(self)
            other_node=Node(other_node)
        elif type(other_node)!=Node:
            other_node=Node(other_node)
        elif type(self)!=Node:
            self=Node(self)
        return self,other_node

def exp(node):
    return Node(np.exp(node.val),parents=[(node,np.exp(node.val))])

def cos(node):
    return Node(np.cos(node.val),parents=[(node,-np.sin(node.val))])

def sin(node):
    return Node(np.sin(node.val),parents=[(node,np.cos(node.val))])

def log(node):
    return Node(np.log(node.val),parents=[(node,1/node.val)])
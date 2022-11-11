# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:14:00 2022

@author: Thomas
"""

import Node as nd
import torch 
import numpy as np

if __name__=="__main__":
    print("Compute the grad of cos(a)²+ln(1+exp(b)) in (4,3)\n")
    a=4.0
    b=3.0
    #With my autodiff
    a_nd = nd.Node(a)
    b_nd = nd.Node(b)
    y_nd = nd.Node(3)*a_nd**3-b_nd**2
    y_nd.backward()
    
    print("grad a =%s,grad b =%s"%(a_nd.grad,b_nd.grad))
    
    a_t = torch.tensor([a], requires_grad=True)
    b_t = torch.tensor([b], requires_grad=True)
    y_t = 3*a_t**3-b_t**2
    y_t.backward()
    
    print("grad a =%s,grad b =%s"%(a_t.grad,b_t.grad))
    


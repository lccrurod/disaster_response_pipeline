# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:46:12 2021

@author: justC
"""

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
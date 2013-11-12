#npr.py
import imageIO as io
#import a2
import numpy as np
import scipy as sp
from scipy import signal
from scipy import ndimage
import random as rnd
import nprHelper as helper
import math

def imIter(im):
 for y in xrange(im.shape[0]):
    for x in xrange(im.shape[1]):
       yield y, x

def brush(out, y, x, color, texture):
    ''' out: the image to draw to.
        y,x: where to draw in out.
        color: the color of the stroke.
        texture: the texture of the stroke.'''
    (height, width, rgb) = texture.shape
    if (y - height/2 < 0) or (y + height/2 >= out.shape[0]) or (x - width/2 < 0) or (x + width/2 >= out.shape[1]):
        return
    y_bottom = y - math.floor(float(height)/2)
    x_bottom = x - math.floor(float(width)/2)
    y_top = y + math.ceil(float(height)/2)
    x_top = x + math.ceil(float(width)/2)
    colored = np.zeros(texture.shape)
    colored += color
    out[y_bottom:y_top, x_bottom:x_top] = (colored * texture) + (out[y_bottom:y_top, x_bottom:x_top] * (1-texture))
    

def singleScalePaint(im, out, importance, texture, size=10, N=1000, noise=0.3):
    '''Paints with all brushed at the same scale using importance sampling.'''
    (height, width, rgb) = out.shape
    scaledTexture = helper.scaleImage(texture, size)
    for step in range(N):
        y=int(rnd.random() * 0.9999 * height)
        x=int(rnd.random() * 0.9999 * width)
        if importance[y,x,0] > rnd.random():
            color_noise = (1 - noise / 2 + noise * np.random.rand(3))        
            brush(out, y, x, im[y,x] * color_noise, scaledTexture)
        

def painterly(im, texture, N=10000, size=50, noise=0.3):
    ''' First paints at a coarse scale using all 1's for importance sampling,
        then paints again at size/4 scale using the sharpness map for importance sampling.'''
    out = np.zeros_like(im)
    singleScalePaint(im, out, np.ones_like(im), texture, float(size), N, noise)
    singleScalePaint(im, out, helper.sharpnessMap(im), texture, float(size)/4, N, noise)
    return out

def computeAngles(im):
    ''' Return an image that holds the angle of the smallest eigenvector of the structure tensor at each pixel.
        If you have a 3 channel image as input, just set all three channels to be the same value theta.'''
    out = np.zeros_like(im)
    tensor = helper.computeTensor(im)
    for y,x in imIter(tensor):
        eigenVecs = np.linalg.eigh(tensor[y,x])[1]
        out[y,x] = min(np.arctan2(eigenVecs[0][0], eigenVecs[0][1]), np.arctan2(eigenVecs[1][0], eigenVecs[1][1]))
    return out

def singleScaleOrientedPaint(im, out, thetas, importance, texture, size, N, noise, nAngles=36):
    '''same as single scale paint but now the brush strokes will be oriented according to the angles in thetas.'''
    (height, width, rgb) = out.shape
    scaledTexture = helper.scaleImage(texture, size)
    rotations = helper.rotateBrushes(texture, nAngles)
    rotations = [helper.scaleImage(t, size) for t in rotations]
    for step in range(N):
        y=int(rnd.random() * 0.9999 * height)
        x=int(rnd.random() * 0.9999 * width)
        if importance[y,x,0] > rnd.random():
            i = nAngles * thetas[y,x] / (2 * math.pi)           
            rotatedTexture = rotations[abs(int(round(i[0])))]
            color_noise = (1 - noise / 2 + noise * np.random.rand(3))        
            brush(out, y, x, im[y,x] * color_noise, rotatedTexture)    

def orientedPaint(im, texture, N=7000, size=50, noise=0.3):
    '''same as painterly but computes and uses the local orientation information to orient strokes.'''
    out = np.zeros_like(im)
    thetas = computeAngles(im)
    singleScaleOrientedPaint(im, out, thetas, np.ones_like(im), texture, float(size), N, noise, 36)
    singleScaleOrientedPaint(im, out, thetas, helper.sharpnessMap(im), texture, float(size)/4, N, noise, 36)
    return out    












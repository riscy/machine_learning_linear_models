#
# Machine learning with linear models - a demo
# Requires pylab to be installed.
#
# Implemented by Chris Rayner (2015)
# dchrisrayner AT gmail DOT com
#
#
# This is a small demo meant to showcase a few simple but different
# linear models for mapping X to Y.  Different assumptions about the
# data can lead to (sometimes drastically) different levels of
# performance.  For instance, when the mapping from X to Y is low rank
# (i.e., an information 'bottleneck'), a technique called reduced rank
# regression (RRR) can outperform standard multivariate linear
# regression (LR).  When the mapping from X to Y is time dependent and
# based on an underlying linear dynamical system, applying a system
# identification technique (SUBID) can result in big gains over LR.
# Have a look at this sample output (or run the program for yourself):
#
# Full Rank Data
# * Multivariate Linear Regression
# Training error: 14.88573 	Testing error: 15.11209     <-- best!
# * Reduced Rank Regression
# Training error: 28.43934 	Testing error: 29.11977
# * Linear Dynamical System
# Training error: 14.9864 	Testing error: 15.23812
# 
# Low Rank Data
# * Multivariate Linear Regression
# Training error: 14.93315 	Testing error: 15.08155
# * Reduced Rank Regression
# Training error: 14.95185 	Testing error: 15.0568      <-- best!
# * Linear Dynamical System
# Training error: 15.84534 	Testing error: 17.14432
# 
# Linear Dynamical System Data
# * Multivariate Linear Regression
# Training error: 107.05471 	Testing error: 107.06858
# * Reduced Rank Regression
# Training error: 122.01291 	Testing error: 120.99966
# * Linear Dynamical System
# Training error: 21.98019 	Testing error: 22.27202     <-- best!
#
#
# My Ph.D. supervisor (Dr. Michael Bowling) introduced me to RRR;
# Dr. Tijl De Bie gave a great talk on subspace system identification
# in 2005 that I modeled my implementation on.
#
#
# Keywords: regression, bottlenecking, system identification
#
# URL: http://www.cs.ualberta.ca/~rayner/
# Git: https://github.com/riscy/mllm/

from pylab import *

import mllmLR
import mllmRRR
import mllmSUBID

# total number of samples
num = 5000

# train/test split
split = 0.5 * num

# dimensionality of input and output
dimX = 20 
dimY = 15

# internal rank
rrank = 10

# noise scale
noise = 1.0

# generate some different kinds of data
Xl,Yl = mllmLR.idealData(num, dimX, dimY, noise)
Xr,Yr = mllmRRR.idealData(num, dimX, dimY, rrank, noise)
Xs,Ys = mllmSUBID.idealData(num, dimX, dimY, rrank, noise)

# regularization
reg = 0

# error function
def sqerr(Yhat, Y):
    return around(pow(norm(Yhat - Y, 'fro'), 2) / size(Y, 0), 5)

# go!
print "\nFull Rank Data"

lr = mllmLR.mllmLR(Xl[:split], Yl[:split], reg)
rrr = mllmRRR.mllmRRR(Xl[:split], Yl[:split], rrank, reg)
subid = mllmSUBID.mllmSUBID(Xl[:split], Yl[:split], rrank, reg)

print lr
print "Training error:", sqerr(lr.predict(Xl[:split]), Yl[:split]),
print "\tTesting error:", sqerr(lr.predict(Xl[split+1:]), Yl[split+1:])
print rrr
print "Training error:", sqerr(rrr.predict(Xl[:split]), Yl[:split]),
print "\tTesting error:", sqerr(rrr.predict(Xl[split+1:]), Yl[split+1:])
print subid
print "Training error:", sqerr(subid.predict(Xl[:split]), Yl[:split]),
print "\tTesting error:", sqerr(subid.predict(Xl[split+1:]), Yl[split+1:])


print "\nLow Rank Data"

lr = mllmLR.mllmLR(Xr[:split], Yr[:split], reg)
rrr = mllmRRR.mllmRRR(Xr[:split], Yr[:split], rrank, reg)
subid = mllmSUBID.mllmSUBID(Xr[:split], Yr[:split], rrank, reg)

print lr
print "Training error:", sqerr(lr.predict(Xr[:split]), Yr[:split]),
print "\tTesting error:", sqerr(lr.predict(Xr[split+1:]), Yr[split+1:])
print rrr
print "Training error:", sqerr(rrr.predict(Xr[:split]), Yr[:split]),
print "\tTesting error:", sqerr(rrr.predict(Xr[split+1:]), Yr[split+1:])
print subid
print "Training error:", sqerr(subid.predict(Xr[:split]), Yr[:split]),
print "\tTesting error:", sqerr(subid.predict(Xr[split+1:]), Yr[split+1:])


print "\nLinear Dynamical System Data"

lr = mllmLR.mllmLR(Xs[:split], Ys[:split], reg)
rrr = mllmRRR.mllmRRR(Xs[:split], Ys[:split], rrank, reg)
subid = mllmSUBID.mllmSUBID(Xs[:split], Ys[:split], rrank, reg)

print lr
print "Training error:", sqerr(lr.predict(Xs[:split]), Ys[:split]),
print "\tTesting error:", sqerr(lr.predict(Xs[split+1:]), Ys[split+1:])
print rrr
print "Training error:", sqerr(rrr.predict(Xs[:split]), Ys[:split]),
print "\tTesting error:", sqerr(rrr.predict(Xs[split+1:]), Ys[split+1:])
print subid
print "Training error:", sqerr(subid.predict(Xs[:split]), Ys[:split]),
print "\tTesting error:", sqerr(subid.predict(Xs[split+1:]), Ys[split+1:])

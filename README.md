# mllm
Machine learning with linear models - a demo
Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Requires pylab to be installed.

This is a small demo meant to showcase a few simple but different
linear models for mapping X to Y.  Different assumptions about the
data can lead to (sometimes drastically) different levels of
performance.  For instance, when the mapping from X to Y is low rank
(i.e., an information 'bottleneck'), a technique called reduced rank
regression (RRR) can outperform standard multivariate linear
regression (LR).  When the mapping from X to Y is time dependent and
based on an underlying linear dynamical system, applying a system
identification technique (SUBID) can result in big gains over LR.
Have a look at this sample output (or run the program for yourself):

Full Rank Data
* Multivariate Linear Regression
Training error: 14.88573 	Testing error: 15.11209     <-- best!
* Reduced Rank Regression
Training error: 28.43934 	Testing error: 29.11977
* Linear Dynamical System
Training error: 14.9864 	Testing error: 15.23812

Low Rank Data
* Multivariate Linear Regression
Training error: 14.93315 	Testing error: 15.08155
* Reduced Rank Regression
Training error: 14.95185 	Testing error: 15.0568      <-- best!
* Linear Dynamical System
Training error: 15.84534 	Testing error: 17.14432

Linear Dynamical System Data
* Multivariate Linear Regression
Training error: 107.05471 	Testing error: 107.06858
* Reduced Rank Regression
Training error: 122.01291 	Testing error: 120.99966
* Linear Dynamical System
Training error: 21.98019 	Testing error: 22.27202     <-- best!

My Ph.D. supervisor (Dr. Michael Bowling) introduced me to RRR;
Dr. Tijl De Bie gave a great talk on subspace system identification
in 2005 that I modeled my implementation on.

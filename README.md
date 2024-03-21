# SynControl

Code associated with the manuscript "Bicknell B. A. & Latham P. E., (2024). Fast 
and slow synaptic plasticity enables concurrent control and learning".

Requirements
-----------------
numpy\
matplotlib\
scipy

Directories
-----------
syn_control: main package for building and running models\
scripts: scripts to run simulations from the paper\
outputs: for saving results

Usage
-----
Example:

To run the teacher-student task with Bayesian control and learning, 
and continuous error feedback, navigate to 'scripts' and use

> python run_task1.py --model Bayes --ftype cts --seed 0

Core model parameters can be edited in syn_control/parameters.py and others 
passed as arguments at the command line.

Note that most simulations in the paper were run remotely, with run times of 
hours to days. For faster simulations, reduce total time T and/or number of 
synapses.

Contact
-------
For any questions please contact B.A.B. via email.
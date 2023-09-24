# Investigating the Evolution of Fisher Information for Neural Network Dynamics
# A bachelor's thesis by Marc Sauter
----
This repository contains all files needed to compile my bachelors thesis with LuaLatex.
Compile the titlepage.tex file seperately first, then compile the main.tex to produce the 
full thesis in "main.pdf".
----

All files needed to create the explanatory plots can be found in the text subfolder,
where the .tex files of the textpassages are located as well.
The files needed to create the plots from the second part of the results section are
located in Experiment2. Experiment1 is a previous version that produced incorrect results.
In the Experiment2 folder you can also find the algorithm to calculate the Fisher Information
and the scalar Curvature using the jax library.
If you want to compile the plots used in the thesis run thesis_plots.py and comment out the
plots you want to be created. If you want to create 3d surfaces of the plots in the thesis,
you can run the plot_experiment.py script and set the corresponding variables at the top of
the file to true.

The plots created in the first part of the results together with the MNIST experiment are too
large for this github repo. But if you read this from the DARUS website, you might find it in
a directory called "Creating data without cluster". Where you can create the plots by executing
the Analyze_condor_data.ipynb notebook. The data from the experiments might be in a directory
called "condor_fisher_data_mypc". Don't worry about the naming of all the files here. They 
probably don't make a lot of sense. Just stick to the files mentioned here.
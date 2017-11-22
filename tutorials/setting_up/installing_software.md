# Table of contents

1. [Introduction](#introduction)
2. [Installing on Windows](#windows)
3. [Installing on Mac](#mac)
4. [Installing on Linux](#linux)

## Introduction 

In this class, we're going to use some special software, including [Python](https://www.python.org/), [TensorFlow](https://www.tensorflow.org/), [Jupyter Notebooks](http://jupyter.org/), and some custom software we've developed to make understanding neural networks easier. This document provides some (hopefully relatively painless) instructions for setting these things up.

## Windows

### Python
Go to <https://www.python.org/downloads/windows/> and download the newest version of Python 3 (3.6.3 as of this writing) executable installer for your operating system. Run the file. Check the box that says "Add python 3.x.x to the path" and then click install now.

To test open your Command Prompt (search "cmd" on the start bar) and run
```
python --version
```
Make sure the version printed is 3.x.x, not 2.x.x.

### TensorFlow

To install [TensorFlow](https://www.tensorflow.org/), run the following command in your command prompt:
```
pip install tensorflow
```
To test the installation, run:
```bash
python -c "\
import tensorflow as tf;\
hello = tf.constant('Hello, TensorFlow!');\
sess = tf.Session();\
print(sess.run(hello));"
```
If you see something saying "Hello, TensorFlow!", you're good to go!

### Jupyter

Install [Jupyter Notebooks](http://jupyter.org/) using pip as follows:

```
pip install jupyter
```

### Our software

To install our software (and also get the homeworks and other class materials), if you use [git](https://git-scm.com/), navigate to the directory where you want to keep your course materials and run
```
git clone https://github.com/alex-ten/pdpyflow.git
```
otherwise download <https://github.com/alex-ten/pdpyflow/archive/master.zip> and extract it in your desired class directory. 

You should be all set now! To test out the last few installs, navigate to your course directory, and then run
```bash
cd pdpyflow; jupyter notebook
```
(If a browser window doesn't open automatically, you may need to copy and paste the link shown.) Use this browser to open some of the .ipynb files in the tutorials/getting_started folder and try them out!

## Mac

TODO

## Linux

We give setup instructions for Ubuntu, but these should be easily generalizable to other Debian-based distros, and with not too much more effort to other distros. 

### Python
Ubuntu (like most modern linux distros) comes with Python 3, but you can check by running
```bash
python3 --version
```
in your terminal.

Install the python package index and development headers by running 
```bash
sudo apt-get install python3-{pip,dev} 
```

### TensorFlow

To install [TensorFlow](https://www.tensorflow.org/), run the following command in your terminal:
```bash
sudo pip3 install tensorflow
```
To test the installation, run:
```bash
python3 -c "\
import tensorflow as tf;\
hello = tf.constant('Hello, TensorFlow!');\
sess = tf.Session();\
print(sess.run(hello));"
```
If you see something saying "Hello, TensorFlow!", you're good to go!

### Jupyter

Install [Jupyter Notebooks](http://jupyter.org/) using pip as follows:

```bash
sudo pip3 install jupyter
```

### Our software

To install our software (and also get the homeworks and other class materials), first navigate to the directory where you will keep class materials, e.g.
```bash
cd ~/Documents/Psych209/
```

If you use [git](https://git-scm.com/), run
```bash
git clone https://github.com/alex-ten/pdpyflow.git
```

otherwise run
```bash
wget https://github.com/alex-ten/pdpyflow/archive/master.zip
unzip master.zip
rm master.zip
mv pdpyflow-master pdpyflow
```

You should be all set now! To test out the last few installs, run
```bash
cd pdpyflow; jupyter notebook
```
(If a browser window doesn't open automatically, you may need to copy and paste the link shown.) Use this browser to open some of the .ipynb files in the tutorials/getting_started folder and try them out!

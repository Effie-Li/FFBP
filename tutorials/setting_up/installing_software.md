# Table of contents

1. [Introduction](#introduction)
2. [Installing on Windows](#windows)
3. [Installing on Mac](#mac)
4. [Installing on Linux](#linux)

## Introduction 

In this class, we're going to use some special software, including [Python](https://www.python.org/), [TensorFlow](https://www.tensorflow.org/), [Jupyter Notebooks](http://jupyter.org/), and some custom software we've developed to make understanding neural networks easier. This document provides some (hopefully relatively painless) instructions for setting these things up. We highly recommend creating a [virtual environment](https://virtualenv.pypa.io/en/stable/) for the class software in order to prevent potential conflicts with whatever is already installed on your computer. A complete list of the required packages can be found [here](https://github.com/alex-ten/pdpyflow/blob/master/tutorials/setting_up/requirements_list.md). Note, that most of these are installed automatically as a result of installing the main software. If you do experience problems, see if they can be solved by switching to a different version of software, as specified in the list. You can check the versions of Python packages with `pip` by running:

```
pip list
```

If you have other difficulties, contact the class TA, Andrew Lampinen, lampinen@stanford.edu for assistance; or (especially if you are using a Mac) contact the developer of the pdpyflow software, Alex Ten, tenalexander1991@gmail.com . 


## Windows

### 1) Python
Download the python 3.6.4 installer from <https://www.python.org/ftp/python/3.6.4/python-3.6.4-amd64.exe>. Run the file. Check the box that says "Add python to the path" and then click install now.

To test open your Command Prompt (search "cmd" on the start bar) and run
```
python --version
```
Make sure the version printed is 3.x.x, not 2.x.x.

### 2) Set up a virtual environment
Install [virtualenv](https://virtualenv.pypa.io/en/stable/) using `pip3` as follows:

```bash
pip3 install virtualenv
```

Once the installation is complete, create a new environment:

```bash
virtualenv %userprofile%\Environments\pdpyflow_env
```

In the example above, a virtual environment named `pdpyflow_env` is added to the `Environments` folder inside the user's home directory. 

Activate the environment by running

```bash
%userprofile%\Environments\pdpyflow_env\bin\activate
```

and you should see the prompt changing to indicate the name of the active environment in parentheses (e.g. `(pdpyflow_env)`). When a virtual environment is activated, `pip` installations are made in the context of this environment. Thus, whenever you want to use the class software, you will need to make sure the associated environment is activated.

To deactivate the environment simply enter `deactivate` in the terminal.

### 3) TensorFlow

To install [TensorFlow](https://www.tensorflow.org/), run the following command in your command prompt (after activating the virtual environment!):
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

### 4) Jupyter

Install [Jupyter Notebooks](http://jupyter.org/), [Scipy](http://scipy.org), and [Matplotlib](http://matplotlib.org/) using pip as follows:

```
pip install jupyter scipy matplotlib
```

To ensure the intended presentation of visualization tools, overwrite the newest version of `ipywidgets` by an older version (`6.0.0`):
```
pip install --upgrade ipywidgets==6.0.0
```

### 5) Our software

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

Mac OS comes with Python 2.7 and pip preinstalled. By downloading and installing a new version of Python (3.6.4) you should end up with mutliple Python interpreters on your machine. This might create confusion when you use pip to install various packages as each Python installation has its own pip executable associated with it. You can check the location(s) of existing interpreters by using `which -a` command, for example

```bash
which -a python
/usr/local/bin
```

### 1) Install Python 
Go to <https://www.python.org/downloads/mac-osx/> and download the newest version of Python 3 (3.6.3 as of this writing) installer for your operating system. Run the file and follow through the steps to complete installation. Note that by default the installation will not override the existing Python builds, so a handle for the new interpreter will be added (`python3` as opposed to `python`). You can check the installation by running

```bash
python3 --version
```

in your terminal. Make sure the version printed is 3.6.3, not 2.7.x. Accordingly, `pip3` will install packages for Python 3.x.x, not the stock version of Python.

### 2) Set up virtual environment
Install [virtualenv](https://virtualenv.pypa.io/en/stable/) using `pip3` as follows:

```bash
sudo pip3 install virtualenv
```

Once the installation is complete, create a new environment and **link it with the correct version** of Python, by providing the `--python` named argument:

```bash
virtualenv --python=$(which python3) ~/Environments/pdpyflow_env
```

In the example above, a virtual environment named `pdpyflow_env` is added to the `Environments` folder inside the user's home directory. 

Activate the environment by running

```bash
source ~/Environments/pdpyflow_env/bin/activate
```

and you should see the prompt changing to indicate the name of the active environment in parentheses (e.g. `(pdpyflow_env)`). When a virtual environment is activated, `pip` installations are made in the context of this environment. Thus, whenever you want to use the class software, you will need to make sure the associated environment is activated. For convenience, you can alias the activation command by adding it to your login shell profile script (see [this page](http://www.dowdandassociates.com/blog/content/howto-set-an-environment-variable-in-mac-os-x-terminal-only/) for more info). Bash is the default login shell on Macs, so you can simply add something like

```
alias actpdp='source ~/Environments/pdpyflow_env/bin/activate'
```

to your `~/.bash_profile` file, then rerun the login script (i.e. run `source ~/.bash_profile` in the terminal) and you should have a `actpdp` shortcut at your disposal. Try entering it in your terminal and see if it activates the environment correctly.

To deactivate the environment simply enter `deactivate` in the terminal.

### 3) Install software requirements within the virtual environment
Make sure virtual environment is activated (see step 2). 

Then, install [TensorFlow](https://www.tensorflow.org/) by running the following command in your terminal:

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

If you see something saying `Hello, TensorFlow!`, you're good to go!

Next install [Jupyter](http://jupyter.org/), [Numpy](http://numpy.org/), [Scipy](http://scipy.org), and [Matplotlib](http://matplotlib.org/) using `pip3` as follows:

```bash
sudo pip3 install jupyter
sudo pip3 install numpy
sudo pip3 install matplotlib
sudo pip3 install scipy
```
To ensure the intended presentation of visualization tools, overwrite the newest version of `ipywidgets` by an older version (`6.0.0`):
```
sudo pip install --upgrade ipywidgets==6.0.0
```


### 4) Our software

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

You need to add a `$PYTHONPATH` variable to your virtual environment source file and set it to be the path to where our software was installed. Run

```bash
echo 'export PYTHONPATH=$PATH:~/Documents/Psych209/pdpyflow' >> ~/Environments/pdpyflow_env/bin/activate
```

which will append `export PYTHONPATH=$PATH:~/Documents/Psych209/pdpyflow` to the activation source file, which will in turn fix the `$PYTHONPATH` variable as needed. (Re)activate the environment to execute path changes (see step 2) and you should be all set. To test out the last few installs, run

```bash
cd pdpyflow; jupyter notebook
```

(If a browser window doesn't open automatically, you may need to copy and paste the link shown.) Use this browser to open some of the .ipynb files in the tutorials/getting_started folder and try them out!

## Linux

We give setup instructions for Ubuntu, but these should be easily generalizable to other Debian-based distros, and with not too much more effort to other distros. 

### 1) Python
Ubuntu (like most modern linux distros) comes with Python 3, but you can check by running
```bash
python3 --version
```
in your terminal.

Install the python package index and development headers by running 
```bash
sudo apt-get install python3-{pip,dev} 
```
### 2) Set up a virtual environment
Install [virtualenv](https://virtualenv.pypa.io/en/stable/) using `pip3` as follows:

```bash
sudo pip3 install virtualenv
```

Once the installation is complete, create a new environment and **link it with the correct version** of Python, by providing the `--python` named argument:

```bash
virtualenv --python=$(which python3) ~/Environments/pdpyflow_env
```

In the example above, a virtual environment named `pdpyflow_env` is added to the `Environments` folder inside the user's home directory. 

Activate the environment by running

```bash
source ~/Environments/pdpyflow_env/bin/activate
```

and you should see the prompt changing to indicate the name of the active environment in parentheses (e.g. `(pdpyflow_env)`). When a virtual environment is activated, `pip` installations are made in the context of this environment. Thus, whenever you want to use the class software, you will need to make sure the associated environment is activated.

If you want to make this easier, you can create a shorter alias for it, by adding the following to your `~/.bashrc` file:

```bash
alias actpdp='source ~/Environments/pdpyflow_env/bin/activate'
```

After creating the alias, you can simply type `actpdp` to activate the environment.

To deactivate the environment simply enter `deactivate` in the terminal.


### 3) TensorFlow

To install [TensorFlow](https://www.tensorflow.org/), run the following command in your terminal (after activating the virtual environment!):
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

### 4) Jupyter and other assorted packages

Install [Jupyter Notebooks](http://jupyter.org/), [Scipy](http://scipy.org), and [Matplotlib](http://matplotlib.org/) using pip as follows:

```bash
sudo pip3 install jupyter matplotlib scipy
```

To ensure the intended presentation of visualization tools, overwrite the newest version of `ipywidgets` by an older version (`6.0.0`):
```
sudo pip3 install --upgrade ipywidgets==6.0.0
```

### 5) Our software

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

-------------------------------------------------------------------------
# Edit Propagation Using Deep Neural Network
##### README (05/10/2018)
-------------------------------------------------------------------------

This code is a simplified implementation of the edit propagation method described in the following paper: 
Yan Gui, Guang Zeng: "Joint Learning of Visual and Spatial Features for Edit Propagation from a Single Image", submitted, May 2018.

### The code is written in Python3.5, and the following packages are used:
1. Tensorflow>=1.5.0. An open source machine learning framework. https://www.tensorflow.org
2. scikit-learn>=0.19.1. Machine learning library in Python. http://scikit-learn.org
3. scikit-image>=0.13.1. Image processing library in Python. http://scikit-image.org
4. OpenCV>=3.3 Image processing library. http://opencv.org 
5. pydensecrf>=1.0. A Python wrapper for Philipp Krähenbühl's Fully-Connected CRFs. https://github.com/lucasb-eyer/pydensecrf

### How to use: 
1. Clone this repository to your local.
2. Run "cd Edita_Propagation_connected" or "cd Edit_Propagation_convolutional"
3. Run "python DP-connected.py" or "python DP-convolutional.py"

If you use GPU for DNN learning, please install the GPU version of TensorFlow, such as "pip install tensorflow-gpu".

You can use this code for scientific purposes only. Use in commercial projects and redistribution are not allowed without author's permission. Please cite (https://github.com/guiyan2018/Edit_Propagation) when using this code. 

====================================================

Personal Contact Information

====================================================

Email:
	(zengzag@hotmail.com)		(Guang Zeng)
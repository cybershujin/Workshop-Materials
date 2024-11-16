# Workshop Materials Guide

This workshop material guide is in three sections:

1. **Overview** of the Workshop format

2. **Pre-requisites and Prep Instructions** for participating in hands-on exercises of the workshop. 

   *<u>IMPORTANT: This is entirely optional</u>* 

   You can still attend the workshop and benefit greatly with the option to complete those exercises on your own. However, many participants prefer to "follow along" during the workshop.

3. **Resources and References** we use in the workshop, organized by section

## Overview:

This workshop is intended to help the cybersecurity professional go from **zero to hero** in Machine Learning, Deep Learning, Artificial Intelligence and Large-Language Model (LLMs), with the goal to be able to apply those computing models to solve cybersecurity challenges. 

This workshop is perfect for **absolute beginners to AI/ML** or those more advanced professionals **looking for hands on instruction on how to choose use cases, models and begin training and implementation**. If you wanted to upskill your cybersecurity career with **practical, actionable skills** in machine learning and AI, this is the workshop for you. **This workshop does not require you to have advanced coding or mathematics skills.** (see the *Pre-requisites and Prep Instructions* for more detail)

**Agenda**

Part 1: AI ML DeepLearning and GenAI - Understanding computing models and what makes AI "artificial intelligence", when to use different computing models based on your use case

Part 2: AI/ML Applications for Cybersecurity - cybersecurity use cases, how to find models and data sets to start experimenting and applying models to your use cases.

Part 3: GenAI Applications for Cybersecurity - GenAI and LLM use cases, the good, the bad and the surprising; RAG and AI Agents

**Goals**

- You will understand the difference between machine learning, deep learning, artificial intelligence, generative artificial intelligence 
- Understand what elements to consider about your use cases to determine the right approach
- Review some examples of good and bad use cases
- Know where to find pre-trained ML and AI models, data sets and how to use them in your org
- How to use GenAI / LLMs
- Learn about the capabilities of RAG and AI Agents

**In This Workshop You Will NOT:**

•Learn how to build a model from scratch (see SANS SEC595 for this type of instruction)

•Do *any* calculus or upper-level math

## Prerequisites and Workshop Prep Instructions

Prerequisites for this workshop are in two sections: the **knowledge** that will help you make the most of the workshop and the **technical preparation** for participating in the hands-on-section of the workshop.

For the technical preparation In this workshop you have the choice to either **Download Anaconda and Jupyter Lab** or you can follow along using the **Cloud Hosted Options** for using the notebooks and models. 

While the instructors recommend that you do the **Download** method, as it will allow you to build familiarity with tools you will use through your Machine Learning and AI journey and allow you to use data without putting that data in a cloud service, we provide instructions for both here. The Download method also allows you to easily take advantage of many of the github projects and Juypter Notebook referenced as part of this workshop. 

Even if you do the **Cloud Hosted** option, you will still have pre-class preparation such as creating accounts on those platforms.

#### Knowledge

**This workshop does not require you to have advanced coding or mathematics skills.** 

It is *helpful* if you have a basic understanding of **Python** and **descriptive** statics (mean, median, mode, average, and standard deviations) and the concept of **regression** (a form of inferential statics) but it is not *required* for you to benefit from the workshop. These skills WILL optimize your understanding of the hands on portion of the workshop. 

My recommendations for a quick refresh on these topics if you need them:

[CodeAcademy's Free Python3 Course](https://try.codecademy.com/learn-python-3) - I am a fan of CodeAcademy generally, I find the subscription worth it. You will find many AI / ML concepts in this platform from BeautifulSoup (important for web scraping to acquire data), to NumPy and Matploitlib courses. 

[CodeCombat](https://codecombat.com/) - This is my **absolute** favorite way to learn coding, and they recently added some AI Learning levels as well! This is a game, that is like a RPG game that is incredibly fun and practical way to learn Python, C++ or basic AI skills.

[YouTube - Descriptive Statistics]([Descriptive Statistics: FULL Tutorial - Mean, Median, Mode, Variance & SD (With Examples)](https://www.youtube.com/watch?v=SplCk-t1BeA)) - mean, median, mode and standard deviations. This is helpful to understand because these concepts are often how we explore data and try to understand "normal" in order to better identify **data anomalies** which is very important to cybersecurity use cases

[YouTube - Linear and Logistic Regression]([Linear Regression vs Logistic Regression | Data Science Training | Edureka](https://www.youtube.com/watch?v=OCwZyYH14uw)) - these are the most implemented use cases in machine learning for cybersecurity applications, because they are based on a use case where you want to **predict or classify.**

#### Technical Prep

**Download Option**

First download this folder and save to your desktop.

2. Get Anaconda. You can do this two ways - download and install Anaconda https://www.anaconda.com/download (preferred, this is what the instructor will use and what the instructions in the ) or use the Anaconda cloud: https://nb.anaconda.cloud/

If you choose to download Anaconda follow these steps for install setup on a Windows Machine:

- Hit next to agree to install

- Accept license agreement click next

- under "Select Installation Type" screen select "Just for me" click next

- under "choose install location" the default is fine, click next

- Make sure the option is NOT SELECTED that says "Add Anaconda to my PATH Environment Variable"

- Register Anaconda3 as your default Python and then install

- Uncheck the box for Anaconda Edition Tutorial and Getting Started with Anaconda (unless you want to) then finish

- Go to your computer's "Start/Search" menu and look for 'anaconda prompt'

- Click the prompt to start the program. The following instructions ALL take place *within the Anaconda Prompt window*.

  1. At the prompt type:

     conda create --name=workshop python=3.10

     (note that the dash before name is two - together with no space)

  2. Hit enter, then type:

     conda activate workshop

  3. Once it finishes, we will install Jupyter Lab. Start this by typing:

     pip install jupyterlab

     (note that this is one word but when you later call the program to start, it is two words!)

  4. Once that finishes, use the CD or Change Directory command within the  to navigate to the workshop folder you downloaded onto your desktop. This will make sure you have access to the files easily when you start the lab

  5. Once you are in the folder, at the prompt type:

     jupyter lab

  6. That should open in your browser, and you should see the Juypter notebooks in your menu. Click on the Dogs vs Cats Keras CNN image classifier. Put your mouse into the first cell of code that is importing libraries. Press the triangle "Play" button at the top. Once this cell completes and the * inside the brackets turns into a number, you are verified successfully prepared.

     

**Cloud Hosted Options**

First download this folder and save to your desktop.

Three steps for cloud hosting, which we recommend you do ALL of these to make sure you can use all of them.

1. For Anaconda Cloud you go to https://nb.anaconda.cloud/ and create an account. Then go to the green circle on the left hand side “Anaconda Toolbox” and Create a New Project. This will allow you to select the files from the workshop folder from your desktop, and leave the environment location as default.
2. Google Colab - Use a google account and make sure you can access https://colab.research.google.com/ you can test this is working properly by visiting [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/) and clicking the "View in Colab" just under the model description details. If you are able to click the Play button that appears once your cursor is in the code box under "Setup" then you are successfully prepared to use this resource.
3. Go to kaggle.com and sign up for an account / sign in with a google account or similar identity provider. Once you have completed that, go to this link [network_intrusion_detection](https://www.kaggle.com/code/orestasdulinskas/network-intrusion-detection) and hit the "Copy & Edit" button in the upper right of the screen and make sure the environment opens.  If you are able to click the Play button that appears once your cursor is in the code box under "Data Cleaning" then you are successfully prepared to use this resource. Note, that this is a large data set you are importing, it will "Spin" a minute before you see an output in a table below it.

## Resources and References

Basic ML and Deep Learning Concepts

[Friendly Machine Learning: Linear Regression and Multiple Line Regression](https://www.kaggle.com/discussions/general/464589)

[5 Types of Neural Networks: An Essential Guide for Analysts](https://datasciencedojo.com/blog/5-main-types-of-neural-networks/)

[Neural Networks: Solving Complex Science Problems](https://www.neilsahota.com/neural-networks-a-solution-for-complex-problems-in-science-and-engineering/)

[Convolutional Neural Network | Deep Learning | Developers Breach](https://developersbreach.com/convolution-neural-network-deep-learning/)

Cats and Captchas

[Google's Artificial Brain Learns to Find Cat Videos | WIRED](https://www.wired.com/2012/06/google-x-neural-network/) and their officical paper [[1112.6209\] Building high-level features using large scale unsupervised learning](https://arxiv.org/abs/1112.6209)

[How to Classify Photos of Dogs and Cats (with 97% accuracy) - MachineLearningMastery.com](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)

Dogs vs Cats Keras CNN image classifier.ipynb - **Download Option** - Available in Workshop folder sourced from from [Github: mohamedamine99/Keras-CNN-cats-vs-dogs-image-classification](https://github.com/mohamedamine99/Keras-CNN-cats-vs-dogs-image-classification) 

[Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/)- **Cloud Hosted Option** 

[Cats or Dogs - using CNN with Transfer Learning](https://www.kaggle.com/code/gpreda/cats-or-dogs-using-cnn-with-transfer-learning)- **Cloud Hosted Option** 

[Building a Cat Detector using Convolutional Neural Networks — TensorFlow for Hackers (Part III) | by ](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)[Venelin](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)[ ](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)[Valkov](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)[ | Medium](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)

[Cats vs Dogs - Part 1 - 92.8% Accuracy - Binary Image Classification with Keras and Deep Learning](https://wtfleming.github.io/blog/keras-cats-vs-dogs-part-1/)

[Captcha Solver – CNN](https://www.kaggle.com/code/matheusparracho/captcha-solver-cnn) - **Cloud Hosted Option** 

[CNN CAPTCHA Solver - 97.8% Accuracy](https://www.kaggle.com/code/tommyott/cnn-captcha-solver-97-8-accuracy) - **Cloud Hosted Option** 

[Solving CAPTCHAs with Convolutional Neural Networks](https://medium.com/@mathparracho/solving-captchas-with-convolutional-neural-networks-89debcc65f55)

Word Embeddings

[Word Embedding Demo: Tutorial](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html)

[Embeddings 101: The foundation of large language models](https://datasciencedojo.com/blog/embeddings-and-llm/)

RAG

[Building a RAG Application in 10 min with Claude 3 and Hugging Face | Medium](https://medium.com/@myscale/building-a-rag-application-in-10-min-with-claude-3-and-hugging-face-10caea4ea293)

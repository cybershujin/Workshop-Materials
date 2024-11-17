# Workshop Materials Guide


### This workshop material guide is in three sections:

1. **Overview** of the Workshop format

2. **Pre-requisites and Prep Instructions** for participating in hands-on exercises of the workshop. 

   *<u>IMPORTANT: This is entirely optional</u>* 

   You can still attend the workshop and benefit greatly with the option to complete those exercises on your own. However, many participants prefer to "follow along" during the workshop.

3. **Resources and References** we use in the workshop, organized by section

## Overview:

This workshop is intended to help the cybersecurity professional go from **zero to hero** in Machine Learning, Deep Learning, Artificial Intelligence and Large-Language Model (LLMs), with the goal to be able to apply those computing models to solve cybersecurity challenges. 

This workshop is perfect for **absolute beginners to AI/ML** or those more advanced professionals **looking for hands on instruction on how to choose use cases, models and begin training and implementation**. If you wanted to upskill your cybersecurity career with **practical, actionable skills** in machine learning and AI, this is the workshop for you. **This workshop does not require you to have advanced coding or mathematics skills.** (see the *Pre-requisites and Prep Instructions* for more detail)

 Whether you're a curious beginner or a seasoned professional, this hands-on, accessible workshop will equip you with the knowledge and skills to leverage AI in your cybersecurity arsenal. No prior AI experience? No worries! We'll take you from zero to hero, ensuring everyone walks away with valuable, applicable insights. We will cover:
"I don't know what AI, Deep Learning and Machine Learning are and at this point I'm too afraid to ask" - an introduction to the differences between these computing types and traditional expert systems
Machine Learning Fundamentals for Cybersecurity Professionals (including a working ML model written in Python you can take away from this course)
Large Language Models (LLMs) in Cybersecurity - Learn how to train and fine-tune LLMs for specific cybersecurity tasks and identify optimal use cases for LLMs, understanding the limitations and potential pitfalls
Navigating the AI Minefield: Pitfalls and Best Practices - choosing the right model or type of AI for the right problem can be challenging, gain knowledge on understanding the strengths and weakness of different AI and ML applications and their cybersecurity applications. You might be surprised at the things AI is actually really bad at!
How to take what you learned here and explore more, recommendations on where to go next to grow your knowledge

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

We do cover HuggingFace in this workshop. Realize that you with need either to have completed the **Download** preparation steps *or* have a Google Cloud or Amazing Cloud account to deploy the model to in that step if you wish to follow along. These accounts are involved setups, and it is *highly* recommended to do this in advance if you wish to follow along.

While the instructors recommend that you do the **Download** method, as it will allow you to build familiarity with tools you will use through your Machine Learning and AI journey and allow you to use data without putting that data in a cloud service, we provide instructions for both here. The Download method also allows you to easily take advantage of many of the github projects and Juypter Notebook referenced as part of this workshop. 

Even if you do the **Cloud Hosted** option, you will still have pre-class preparation such as creating accounts on those platforms.



### -------- Knowledge --------

**This workshop does not require you to have advanced coding or mathematics skills. There is no requirement to attend other than a basic understanding of cybersecurity concepts.** 

It is *helpful* if you have a basic understanding of **Python** and **descriptive** statics (mean, median, mode, average, and standard deviations) and the concept of **regression** (a form of inferential statics) but it is not *required* for you to benefit from the workshop. These skills WILL optimize your understanding of the hands on portion of the workshop. 

My recommendations for a quick refresh on these topics if you need them:

[CodeAcademy's Free Python3 Course](https://try.codecademy.com/learn-python-3) - I am a fan of CodeAcademy generally, I find the subscription worth it. You will find many AI / ML concepts in this platform from BeautifulSoup (important for web scraping to acquire data), to NumPy and Matploitlib courses. 

[AI Python for Beginners - DeepLearning.AI](https://www.deeplearning.ai/short-courses/ai-python-for-beginners/) - this instructor is great and teaches a number of machine learning concepts, the site itself is a wonderful resource for the beginner!

[CodeCombat](https://codecombat.com/) - This is my **absolute** favorite way to learn coding, and they recently added some AI Learning levels as well! This is a game, that is like a RPG game that is incredibly fun and practical way to learn Python, C++ or basic AI skills.

[YouTube - Descriptive Statistics](https://www.youtube.com/watch?v=SplCk-t1BeA) - mean, median, mode and standard deviations. This is helpful to understand because these concepts are often how we explore data and try to understand "normal" in order to better identify **data anomalies** which is very important to cybersecurity use cases

[YouTube - Linear and Logistic Regression](https://www.youtube.com/watch?v=OCwZyYH14uw) - these are the most implemented use cases in machine learning for cybersecurity applications, because they are based on a use case where you want to **predict or classify.**

Bonus - [Introduction to Data Science]([Introduction to Data Science](https://learning.anaconda.cloud/introduction-to-data-science)) from Anaconda

### -------- Technical Prep --------

Preferably, participants will have the following prerequisites to make the most out of the class - but this is not required. This just allows people in class to follow along with practical, hands-on work:
- Admin rights to install software
- Device with at least 16 GB RAM
- At least 200GB free space
- NVIDIA RTX series (for optimal performance) - at least 4 GB VRAM
- Python 3.7 or higher installed
- ChatGPT Plus subscription ($20) and/or Claude.ai Pro or Team Plan (we will cover both of these in class)
- Ability to reach other AI sites such as perplexity, gamma.app, HuggingFace, Keras,io and Google Colab

  To prepare yourself for the hands-on portion of the class, you must choose between the **Download Option** or the **Cloud Hosted Option** instructions below.
  Instructor generally recommends the download option, because this will best allow you to gain skills you will need to using these models with non-public data sets. However, the **Cloud Hosted Option** is a much faster setup.

**Download Option**

First download this folder from this github and save to your desktop.

Sign up for a Kaggle Account. https://www.kaggle.com/ you will need this in order to follow along with the some of the model exploration parts of the class.

Download the nslkdd data set for the Network Instrusion (Anomaly Detection) exercise from [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) and make sure it is in the same saved in the same workshop folder. (This becomes important when you open Juypter Lab)

Get Anaconda. You can do this two ways - download and install Anaconda https://www.anaconda.com/download 

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

First download this folder from this github and save to your desktop.

Three options for cloud hosting, which we recommend you do ALL of these to make sure you can use all of them with the workshop materials provided below.

However, there is only **two** we will go through together in the workshop - Kaggle and HuggingFace. The Ananconda Cloud and Google Colab are also options for using you could use to host the same notebooks and data and will allow you to follow along.

**To follow along in the workshop** Go to kaggle.com and sign up for an account / sign in with a google account or similar identity provider. Once you have completed that, go to this link [network_intrusion_detection](https://www.kaggle.com/code/orestasdulinskas/network-intrusion-detection) and hit the "Copy & Edit" button in the upper right of the screen and make sure the environment opens.  If you are able to click the Play button that appears once your cursor is in the code box under "Data Cleaning" then you are successfully prepared to use this resource. Note, that this is a large data set you are importing, it will "Spin" a minute before you see an output in a table below it.

If you wish to follow along with the HuggingFace instruction have a Google Cloud or Amazing Cloud account to deploy the model to in that step if you wish to follow along. Go to [Huggingface.co](https://huggingface.co) and sign up for an account. You will know you have met the prerequisites for this if you can go to [ehsanaghaei/SecureBERT · Hugging Face](https://huggingface.co/ehsanaghaei/SecureBERT) and click "Deploy" and select your platform.

Other options which will also work with Jupyter Notebook files (primarily used in this workshop):
For Anaconda Cloud you go to https://nb.anaconda.cloud/ and create an account. Then go to the green circle on the left hand side “Anaconda Toolbox” and Create a New Project. This will allow you to select the files from the workshop folder from your desktop, and leave the environment location as default.

   Quick link [Anaconda Cloud](https://anaconda.cloud/code-in-the-cloud)

Google Colab - Use a google account and make sure you can access https://colab.research.google.com/ you can test this is working properly by visiting [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/) and clicking the "View in Colab" just under the model description details. If you are able to click the Play button that appears once your cursor is in the code box under "Setup" then you are successfully prepared to use this resource.
   

## Resources and References

**Network Intrusion** **Model Used in the Workshop:**

[Intrusion Detection System with ML&DL](https://www.kaggle.com/code/essammohamed4320/intrusion-detection-system-with-ml-dl) - **Example we use in class** - **Cloud Hosted Option**

[Intrusion Detection System NSL-KDD\]](https://www.kaggle.com/code/eneskosar19/intrusion-detection-system-nsl-kdd) is another - **Cloud Hosted Option**

intrusion-detection-system-with-ml-dl.ipynb - **Example we use in class** - **Download Option** - Available in Workshop folder sourced from from

### Other learning resources:

[PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity: Hands-On Artificial Intelligence for Cybersecurity, publised by Packt](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)

[Network Traffic Anomaly Detection with Machine Learning](https://eyer.ai/blog/network-traffic-anomaly-detection-with-machine-learning/)

### Cybersecurity Datasets:

[gfek/Real-CyberSecurity-Datasets: Public datasets to help you address various cyber security problems.](https://github.com/gfek/Real-CyberSecurity-Datasets)

[BNN-UPC/](https://github.com/BNN-UPC/NetworkModelingDatasets)[NetworkModelingDatasets](https://github.com/BNN-UPC/NetworkModelingDatasets)[: This repository contains datasets for network modeling simulated with ](https://github.com/BNN-UPC/NetworkModelingDatasets)[OMNet](https://github.com/BNN-UPC/NetworkModelingDatasets)[++](https://github.com/BNN-UPC/NetworkModelingDatasets)

[ericyoc](https://github.com/ericyoc/synthetic_network_traffic_simulation_poc)[/](https://github.com/ericyoc/synthetic_network_traffic_simulation_poc)[synthetic_network_traffic_simulation_poc](https://github.com/ericyoc/synthetic_network_traffic_simulation_poc)[: A simulation of network traffic using synthetic network traffic for 802.11, 3G GSM, 4G LTE, and 5G NR](https://github.com/ericyoc/synthetic_network_traffic_simulation_poc)

Endpoint telemetry datasets

[ScarredMonk](https://github.com/ScarredMonk/SysmonSimulator)[/](https://github.com/ScarredMonk/SysmonSimulator)[SysmonSimulator](https://github.com/ScarredMonk/SysmonSimulator)[: Sysmon event simulation utility which can be used to simulate the attacks to generate the Sysmon Event logs for testing the EDR detections and correlation rules by Blue teams.](https://github.com/ScarredMonk/SysmonSimulator)

[tsale](https://github.com/tsale/EDR-Telemetry)[/EDR-Telemetry: This project aims to compare and evaluate the telemetry of various EDR products.](https://github.com/tsale/EDR-Telemetry)

### Beginner Friendly AI/ML Cybersecurity Models:

Captchas

[Captcha Solver – CNN](https://www.kaggle.com/code/matheusparracho/captcha-solver-cnn) - **Cloud Hosted Option** and its accompanying blog [Solving CAPTCHAs with Convolutional Neural Networks | by Matheus Ramos Parracho | Medium](https://medium.com/@mathparracho/solving-captchas-with-convolutional-neural-networks-89debcc65f55)

[CNN CAPTCHA Solver - 97.8% Accuracy](https://www.kaggle.com/code/tommyott/cnn-captcha-solver-97-8-accuracy) - **Cloud Hosted Option** 

[Solving CAPTCHAs with Convolutional Neural Networks](https://medium.com/@mathparracho/solving-captchas-with-convolutional-neural-networks-89debcc65f55)

Network Threat Detection / Anomaly Detection / Intrusion Analysis

[How to do Anomaly Detection using Machine Learning in Python?](https://www.projectpro.io/article/anomaly-detection-using-machine-learning-in-python-with-example/555)

[Intrusion Detection System with ML&DL](https://www.kaggle.com/code/essammohamed4320/intrusion-detection-system-with-ml-dl/notebook)  - machine learning

[Network Traffic Anomaly Detection](https://www.kaggle.com/code/vidhikishorwaghela/network-traffic-anomaly-detection) - deep learning model

[yasakrami/Threat-Detection-in-Cyber-Security-Using-AI](https://github.com/yasakrami/Threat-Detection-in-Cyber-Security-Using-AI) Using PCAP files

Spam vs Ham (and learning about unbalanced data sets)

[HAM vs SPAM Email Classifier (CountVect & TF-IDF)](https://www.kaggle.com/code/jacopoferretti/ham-vs-spam-email-classifier-countvect-tf-idf)

More advanced:

[Algorithmic-Machine-Learning/Lecture 9+10\] Anomaly Detection in Network Traffic with K-means clustering.ipynb at master · lucabenedetto/Algorithmic-Machine-Learning](https://github.com/lucabenedetto/Algorithmic-Machine-Learning/blob/master/[Lecture 9%2B10] Anomaly Detection in Network Traffic with K-means clustering.ipynb) - indepth, covers different ways to find outliers and anomalies using supervised and unsupervised machine learning

### Language models:

[llama-recipes/recipes/quickstart at main · meta-llama/llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart)

**Cybersecurity Domain-Specific Language Model**

[SynamicTechnologies/CYBERT · Hugging Face](https://huggingface.co/SynamicTechnologies/CYBERT)

[ehsanaghaei/SecureBERT: SecureBERT is a domain-specific language model to represent cybersecurity textual data.](https://github.com/ehsanaghaei/SecureBERT) and [ehsanaghaei/SecureBERT · Hugging Face](https://huggingface.co/ehsanaghaei/SecureBERT)

[markusbayer/CySecBERT · Hugging Face](https://huggingface.co/markusbayer/CySecBERT)

**Generative AI and LLMs for the Cybersecurity Professional:**

[Prompt Engineering | Lil'Log](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) - Getting started understanding prompt engineering

https://microsoft.github.io/prompt-engineering/ - Prompt Engineering for Code

https://github.com/promptslab/Awesome-Prompt-Engineering?tab=readme-ov-file#tools--code - Prompt Engineering

https://library.easyprompt.xyz/?via=topaitools - Prompt Library

https://cloud.google.com/blog/topics/threat-intelligence/ai-nist-nice-prompt-library-gemini NIST NICE Prompt Library

https://github.com/Billy1900/Awesome-AI-for-cybersecurity

https://github.com/DummyKitty/Cyber-Security-chatGPT-prompt - Cybersecurity prompt library

https://github.com/fr0gger/Awesome-GPT-Agents

https://chatgpt.com/g/g-2DQzU5UZl-code-copilot 

https://chatgpt.com/g/g-jBdvgesNC-diagrams-flowcharts-mindmaps 

https://github.com/tenable/awesome-llm-cybersecurity-tools 

https://github.com/JusticeRage/Gepetto <- for use with IDA Pro for malware reverse engineering assistance 

https://github.com/s0md3v/SubGPT < subdomain enumeration 

https://chatgpt.com/g/g-IZ6k3S4Zs-mitregpt <- MITRE ATT&CK mapping 

https://github.com/Mooler0410/LLMsPracticalGuide

**Anaconda and Data Science:**

[An End-to-end Data Science Project with Anaconda Assistant | Anaconda](https://www.anaconda.com/blog/end-to-end-data-science-project-with-anaconda-assistant)

### Basic ML and Deep Learning Concepts:

[Friendly Machine Learning: Linear Regression and Multiple Line Regression](https://www.kaggle.com/discussions/general/464589)

[5 Types of Neural Networks: An Essential Guide for Analysts](https://datasciencedojo.com/blog/5-main-types-of-neural-networks/)

[Neural Networks: Solving Complex Science Problems](https://www.neilsahota.com/neural-networks-a-solution-for-complex-problems-in-science-and-engineering/)

[Convolutional Neural Network | Deep Learning | Developers Breach](https://developersbreach.com/convolution-neural-network-deep-learning/)

**Cats and Captchas**

[Google's Artificial Brain Learns to Find Cat Videos | WIRED](https://www.wired.com/2012/06/google-x-neural-network/) and their officical paper [[1112.6209\] Building high-level features using large scale unsupervised learning](https://arxiv.org/abs/1112.6209)

[How to Classify Photos of Dogs and Cats (with 97% accuracy) - MachineLearningMastery.com](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)

Dogs vs Cats Keras CNN image classifier.ipynb - **Download Option** - Available in Workshop folder sourced from from [Github: mohamedamine99/Keras-CNN-cats-vs-dogs-image-classification](https://github.com/mohamedamine99/Keras-CNN-cats-vs-dogs-image-classification) 

[Cat & Dog Classification using Convolutional Neural Network in Python - GeeksforGeeks](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/#) for use with **Download Option** 

[Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/)- **Cloud Hosted Option** 

[Cats or Dogs - using CNN with Transfer Learning](https://www.kaggle.com/code/gpreda/cats-or-dogs-using-cnn-with-transfer-learning)- **Cloud Hosted Option** 

[Building a Cat Detector using Convolutional Neural Networks — TensorFlow for Hackers (Part III) | by ](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)[Venelin](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)[ ](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)[Valkov](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)[ | Medium](https://venelinvalkov.medium.com/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)

[Cats vs Dogs - Part 1 - 92.8% Accuracy - Binary Image Classification with Keras and Deep Learning](https://wtfleming.github.io/blog/keras-cats-vs-dogs-part-1/)

[Captcha Solver – CNN](https://www.kaggle.com/code/matheusparracho/captcha-solver-cnn) - **Cloud Hosted Option** and its accompanying blog [Solving CAPTCHAs with Convolutional Neural Networks | by Matheus Ramos Parracho | Medium](https://medium.com/@mathparracho/solving-captchas-with-convolutional-neural-networks-89debcc65f55)

[CNN CAPTCHA Solver - 97.8% Accuracy](https://www.kaggle.com/code/tommyott/cnn-captcha-solver-97-8-accuracy) - **Cloud Hosted Option** 

[Solving CAPTCHAs with Convolutional Neural Networks](https://medium.com/@mathparracho/solving-captchas-with-convolutional-neural-networks-89debcc65f55)

**Word Embeddings**

[Word Embedding Demo: Tutorial](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html)

[Embeddings 101: The foundation of large language models](https://datasciencedojo.com/blog/embeddings-and-llm/)

**Data Science Process**

[Data Science Process: A Beginner’s Guide in Plain English](https://www.springboard.com/blog/data-science/data-science-process/)

[Introduction to Data Science]([Introduction to Data Science](https://learning.anaconda.cloud/introduction-to-data-science)) from Anaconda

[Data Science Process: 7 Steps With Comprehensive Case Study](https://www.embedded-robotics.com/data-science-process/)

**RAG**

[Building a RAG Application in 10 min with Claude 3 and Hugging Face | Medium](https://medium.com/@myscale/building-a-rag-application-in-10-min-with-claude-3-and-hugging-face-10caea4ea293)




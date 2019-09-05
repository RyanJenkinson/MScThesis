# :robot:TraMML - Transformers with Multitask and Meta Learning
Ryan Jenkinson, MSc Project (2019) - An investigation into multitask learning and meta learning for NLU problems, with specific focus on Aspect Based Sentiment Analysis. This project is in industry collaboration with Streetbees.

This project will investigate a multitask learning approach to aspect based sentiment analysis, and investigate whether auxilliary tasks improve the performance of our base language models (which are Transformers) by supporting the primary task using various defined sampling schemas.

This project relies heavily on the amazing work done by the ```pytorch-transformers``` library by HuggingFace (https://github.com/huggingface/pytorch-transformers), and we build additional functionality on top of their models that allow users to easily fine tune their models, but also joint multitask fine tune different tasks.

# How to use this Library
All of our data is in a standard format. We create a csv file with either 3 or four columns. The first column is the ID, which is used to generate the unique id in the ```extract_features.py``` script. Then we have a column corresponding to ```text_a```, the first sentence (Sentence A) ingested by our language model. If our task requires an additional sentence (Sentence B), then we create a column called ```text_b```. The ```label``` column has the corresponding label or, in the case of the NER task, a list of labels.

For any new dataset, create a ```.csv``` file as per these examples, and add a DataProcessor object in ```data_processing.py``` (remembering to add it to the PROCESSORS object). One can modify the ```run_config.json``` in ```multitask_learning/configs/``` to run ```learners.py``` or set up some experiments as in ```run_experiments.py``` to test out different sampling modes.

The arguments to the run config can be found in the initialisation of the MultiTaskLearner in ```learners.py```, as well as all the defaults.

This library is very flexible, and can be used to easily fine tune SOTA language models on any task, as well as any more complicated multitask learning required.

# Meta Learning [WIP]
The goal was to also introduce a stable way to meta learn word embeddings using the Reptile algorithm to improve generalisation to new tasks, but it appears to be unstable at this current version, and does not learn effectively. 

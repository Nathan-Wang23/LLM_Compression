# Code Structure:

* [Checkpoints](./checkpoints/) - This contains our model checkpoints for both baselines and novel, each of which are .pt files.
* [Dataset Definition](./dataset_definitions/) - A place to add multiple different dataset classes (different class logic may be needed for different datasets. Since we do perplexity for the training dataset (on data not seen by the model), we only provide that dataset definition for now)
* [eval](./eval/) - This is adapted from a framework named TRACE.
* [generators](./generators/) - We adapt the generator API from Llama here.
* [logs](./logs/) - The relevant logs we provide are:
    * [losses](./logs/losses/) - These are our loss details, which are **encoded in binary**. They are not human readable easily. We use pickle for dumping binary into the file. To see the meaning of the data encoded, you can refer to how we decode this in the visualizations directory (e.g. for the [training curve](../../visualizations/viz.ipynb)), and how we encode this in our [trainer file](./generic_runner.py)
    * [times](./logs/times/) - We log the elapsed training time for each run. Do note these are again subject to bias because the GPU may have been in use for evaluation as well, causing potential slowdowns.
    * [perplexities](./logs/perplexity-losses/) - These are **lists** of exponentiated values. We do averaging (np.mean, how perplexity is calculated) in our visualizations of perplexity.
* Model definitions:
    * [Non-KV Models](./models/) - These are model configurations that can directly be used in training.
    * [KV Cache Models](./modelskv/) - These are model configuration that are better optimized for inference.
* [Perplexity Runners](./perplexity-runners) - This contains a script for calculating perplexities of the models across 10K unseen rows of data from the original training dataset.
* [Tokenizers](./tokenizers/) - The original, pre-trained Llama tokenizer utilized by Meta.
* [Generic Trainer Utility](./generic_runner.py) - This contains our entire entrypoint into trianing a model. We set up a distributed environment, load our dataset, create a new model, and train for approximately 46,000 batches of size 32, or around 1.5M rows.
* Runner files
    * We have 8+ runner files. Simply executing the file in python starts the whole training process, which uses the generic trainer utility file. Be sure to change save, losses, and times path to nonexistent paths (we have assertions to prevent overwrites).
* Evaluation files by Uzair
    * These are a series of Jupyter notebooks in this directory to generates outputs across different datasets (E.g. Antrophic's dataset) from our trained models.

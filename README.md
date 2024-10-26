# GECToR Lightning
A PyTorch Lightning-based re-implementation of the [GECToR](https://github.com/grammarly/gector) model from Grammarly.

## Installation

1. Clone the repo.
2. Run:
    ```bash
    poetry install
    ```

## Training & Post-Training Steps
### Prepare Data
Use scripts under `scripts/data_processing/per_dataset_scripts` to convert data into appropriate format (GECToR format 
for training & validation sets, ERRANT-annotated M2 for the test set). See the original [repo](https://github.com/grammarly/gector) & [paper](https://arxiv.org/abs/2005.12592) for instructions on how to download the data.

Training consists of 3 stages, as outlined in the paper. We used the following tags to indicate for which purpose each dataset is used (train stage X, dev, test):
- **<span style="color: #1E90FF;">[Train 1]</span>**: Used in the first training stage.
- **<span style="color: #32CD32;">[Train 2]</span>**: Used in the second training stage.
- **<span style="color: #9370DB;">[Train 3]</span>**: Used in the third training stage.
- **<span style="color: #FFA500;">[Dev]</span>**: Used for validation.
- **<span style="color: #20B2AA;">[Test]</span>**: Used for testing.

#### 1. PIE 
- Synthetic data consisting of 5 subsets. Only 1 subset might suffice. Can be used as a **train** set.
- Run the following script to process the data:
    ```bash
    ./process_pie.sh <pie_directory>  # e.g. ./process_pie.sh /path/to/synthetic
    ```
  - `<pie_directory>` should be the directory containing the `a1`, ..., `a5` directories.
- Output files: **<span style="color: #1E90FF;">[Train 1]</span>**
  - `<pie_directory>/gector/a1.txt`
  - ...
  - `<pie_directory>/gector/a5.txt`

#### 2. FCE v2.1
- Contains train, dev & test sets, but they will be concatenated to be used as a **training** set.
- Run the following script to process the data:
    ```bash
    ./process_fce.sh <fce_directory>  # e.g. ./process_fce.sh /path/to/fce_v2.1.bea19/fce
    ```
  - `<fce_directory>` should be the directory containing the `m2` directory.
- Output file: `<fce_directory>/gector/fce.txt` **<span style="color: #32CD32;">[Train 2]</span>**

#### 3. Lang-8 Corpus of Learner English
- **Training** set.
- Run the following script to process the data:
    ```bash
    ./process_lang8.sh <lang8_directory>  # e.g. ./process_lang8.sh /path/to/lang8.bea19
    ```
  - `<lang8_directory>` should be the directory containing the `lang8.train.auto.bea19.m2` file.
- Output file: `<lang8_directory>/gector/lang8.txt` **<span style="color: #32CD32;">[Train 2]</span>**

#### 4. NUCLE
- **Training** set.
- Run the following script to process the data:
    ```bash
    ./process_nucle.sh <nucle_directory>
    ```
  - `<nucle_directory>` should be the directory containing the `bea2019` directory.
- Output file: `<nucle_directory>/gector/nucle.txt` **<span style="color: #32CD32;">[Train 2]</span>**

#### 5. W&I+LOCNESS v2.1
- A high quality dataset. Contains both **training** and **validation** sets.
- Run the following script to process the data:
    ```bash
    ./process_wi+locness.sh <wi+locness_directory>  # e.g. ./process_wi+locness.sh /path/to/wi+locness_v2.1.bea19/wi+locness
    ```
  - `<wi+locness_directory>` should be the directory containing the `m2` directory.
- Output files: 
  - `<wi+locness_directory>/gector/wi+locness_train.txt` **<span style="color: #32CD32;">[Train 2]</span>** **<span style="color: #9370DB;">[Train 3]</span>**
  - `<wi+locness_directory>/gector/wi+locness_dev.txt`  **<span style="color: #FFA500;">[Dev]</span>**
    
#### 6. CoNLL-14
- **Test** set to be re-annotated with ERRANT in M2 format.
- Run the following script to process the data:
    ```bash
    ./process_conll14_test.sh <conll14_directory>  # e.g. ./process_conll14_test.sh /path/to/conll14st-test-data
    ```
  - `<conll14_directory>` should be the directory containing the `alt` directory.
- Output file: `<conll14_directory>/conll14-errant-auto.m2` **<span style="color: #20B2AA;">[Test]</span>**

### Train
Training script: `scripts/training/train.py`.
To simulate the 3-stage training process mentioned in the paper, you should follow these steps:
1. 
   1. Copy the `conf_examples/train/stage1.yaml` and modify it for your environment. The most important parameters to modify are:
       - `data_module.train.dataset_params`: Set to the path of the **<span style="color: #1E90FF;">[Train 1]</span>** set.  
         Either:
         - ```yaml
           path: /path/to/pie/gector/a1.txt
           ```
         Or:
         - ```yaml
           paths:
             - /path/to/pie/gector/a1.txt
             - /path/to/pie/gector/a2.txt
             - ...
           ```
       - `data_module.val.dataset_params.path`:  Set to the path of the **<span style="color: #FFA500;">[Dev]</span>** set (path to `gector/wi+locness_dev.txt`.
       - `model.output_vocab`: You can change `num_tokens` and `path` if you have a custom output vocabulary. Currently, is set to the default vocabulary taken from the original repo.
       - `batch_size`: Set to a value that fits your GPU memory.
       - `trainer.accumulate_grad_batches`: Set considering the batch size, GPU memory, and the number of GPUs.
       - `trainer.limit_train_batches`: Again, you can change this based on your `batch_size` and `accumulate_grad_batches` settings. The current value of 160,000 reflects the number of steps of 10,000 (taken from the original repo) with 16 accumulation steps.
       - `trainer.max_epochs`: Currently set to -1, which means the training will continue until manually stopped. You can adjust this if you want to train for a specific number of epochs.
       - You can check the other parameters in the file and modify them as needed.
   2. Run the training script:
      ```bash
      python scripts/train/train.py --config-path /path/to/your/stage1.yaml --experiment-dir /path/to/save/experiments
      ```
      You can check the training progress by running `tensorboard` on the experiment directory.
2. 
   1. Copy the `conf_examples/train/stage2.yaml` and modify it like the previous step. Additionally, you should set the `init_from_pretrained_model` parameter to the path of the best model from the previous stage (choose from `checkpoints` directory under your the experiment directory).
   2. Run the training script:
      ```bash
      python scripts/train/train.py --config-path /path/to/your/stage2.yaml --experiment-dir /path/to/save/experiments
      ```
3. Repeat the same process as step 2 for step 3 (using the best model from step 2) and run the training script again:
   ```bash
   python scripts/train/train.py --config-path /path/to/your/stage3.yaml --experiment-dir /path/to/save/experiments
   ```

### Test
#### Inference Parameter Tuning
1. Copy the `conf_examples/inference.yaml` and modify it for your environment, based on the settings you used in your training configs.
2. Set the `init_from_pretrained_model` parameter to the path of the best model from the last training stage. 
3. Run `scripts/train/inference.py` script with the `grid-search` command. Example:
   ```bash
   python scripts/train/inference.py grid-search --input-path /path/to/m2/test/data --config-path /path/to/your/inference.yaml --device cuda --num-samples-per-param 5
   ```
   - `/path/to/m2/test/data` should be the path to the **<span style="color: #20B2AA;">[Test]</span>** set (path to `conll14-errant-auto.m2`).
4. Best parameters will be printed on the console. You can copy them to your `inference.yaml` file to use it in further testing or production.

#### Evaluate
To test the model on the **<span style="color: #20B2AA;">[Test]</span>** set with the parameters found by grid search, you should run the `scripts/train/inference.py` script with the `evaluate` command. Example:
```bash
python scripts/train/inference.py evaluate --input-path /path/to/m2/test/data --config-path /path/to/your/inference.yaml --device cuda
```
PLACE: Prompt Learning for Attributed Community Search
-----------------
A PyTorch + torch-geometric implementation of PLACE, as described in the paper: Shuheng Fang, Kangfei Zhao, Rener Zhang, Yu Rong, Jeffrey Xu Yu. [PLACE: Prompt Learning for Attributed Community Search]


### Requirements
```
python 3.8
networkx
numpy
scipy
scikit-learn
torch 1.12.1
torch-geometric 2.0.0
```

Import the conda environment by running
```
conda env create -f PLACE.yml
conda activate PLACE
```


### Quick Start
Running cornell
```
python main.py    \
       --data_set cornell
```

### Key Parameters
All the parameters with their default value are in main.py

| name | type   | description |
| ----- | --------- | ----------- |
| num_layers  | int    | number of GNN layers    |
| gnn_type | string |  type of GNN layer (GCN, GAT, RGCN)     |
| total_epoch  | int   | number of training epochs  |
| training_size  | int   | number of training queries |
| test_size  | int   | number of test queries |
| total_query  | int   | total number of queries |
| dataset  | string   | dataset |
| num_pos  | float   | maximum proportion of positive instances for each query node |
| num_neg  | float   | maximum proportion of negative instances for each query node |



### Project Structure
```
main.py         # begin here
load_data.py         # process data
QueryDatasetAttr.py  # generate queries
eva/CSAPeva.py                       # train, valid and test for PLACE
models/QueryPrompt.py                      # model for prompt graph of PLACE
models/CSAttrPrompt.py                     # ACS model of PLACE
```
The Amazon-2m datasets are from [OGB](https://ogb.stanford.edu/docs/nodeprop/);
The WeBKB datasets are from (https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/);
The Cora/Citeseer/Reddit datasets are from PyTorch_Geometric;
The orkut datasets are from [SNAP] (https://snap.stanford.edu/data/com-Orkut.html).


### Contact
Open an issue or send email to shfang@se.cuhk.edu.hk if you have any problem

### Cite Us


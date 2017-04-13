pre_process:
	python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo -domain_train_src data/IWSLT/train.de-en.en -domain_valid_src data/IWSLT/valid.de-en.en

train:
	python train.py -data data/demo-train.pt -save_model model -adapt -learning_rate 0.0001

pre_process_augmented:
	python domain_preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -test_src data/src-test.txt -test_tgt data/tgt-test.txt -save_data data/demo -domain_train_src data/IWSLT/train.de-en.en -domain_valid_src data/IWSLT/valid.de-en.en -domain_test_src data/IWSLT/test.de-en.en -src_vocab data/all.src.dict

pre_process:
	python domain_preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -test_src data/src-test.txt -test_tgt data/tgt-test.txt -save_data data/demo -domain_train_src data/IWSLT/train.de-en.en -domain_valid_src data/IWSLT/valid.de-en.en -domain_test_src data/IWSLT/test.de-en.en

merge:
	python domain_preprocess.py -train_src data/all_src.txt -train_tgt data/all_src.txt -valid_src data/all_src.txt -valid_tgt data/all_src.txt -test_src data/all_src.txt -test_tgt data/all_src.txt -save_data data/all -src_vocab_size 80000 -tgt_vocab_size 1 
train:
	python train.py -data data/demo-train.pt -save_model model -adapt -learning_rate 0.0001 

train_g:
	python train.py -data data/demo-train.pt -save_model model -adapt -learning_rate 0.0001 -gpu 0

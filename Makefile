IWSLT_TRAIN_SRC=data/IWSLT/train.de-en.en
IWSLT_TRAIN_TRG=data/IWSLT/train.de-en.de
IWSLT_VALID_SRC=data/IWSLT/valid.de-en.en
IWSLT_VALID_TRG=data/IWSLT/valid.de-en.de
IWSLT_TEST_SRC=data/IWSLT/test.de-en.en
IWSLT_TEST_TRG=data/IWSLT/test.de-en.de
IWSLT_DATA=data/IWSLT/demo
IWSLT_DATA_PT=data/IWSLT/demo.train.pt
IWSLT_SAVE_MODEL=iwslt_model
IWSLT_OUTPUT=iwslt_pred.txt

WMT14_TRAIN_SRC=data/src-train.txt
WMT14_TRAIN_TRG=data/tgt-train.txt
WMT14_VALID_SRC=data/src-val.txt
WMT14_VALID_TRG=data/tgt-val.txt
WMT14_TEST_SRC=data/src-test.txt
WMT14_TEST_TRG=data/tgt-test.txt
WMT14_DATA=data/demo
WMT14_DATA_PT=data/demo.train.pt
WMT14_SAVE_MODEL=wmt14_model
WMT14_OUTPUT=wmt14_pred.txt

TRAIN_SRC=data/multi30k/train.en.atok
TRAIN_TRG=data/multi30k/train.de.atok
VALID_SRC=data/multi30k/val.en.atok
VALID_TRG=data/multi30k/val.de.atok
TEST_SRC=data/multi30k/test.en.atok
TEST_TRG=data/multi30k/test.de.atok
DATA=data/multi30k.atok
DATA_PT=data/multi30k.atok-train.pt
SAVE_MODEL=wmt16_multi30k_model
OUTPUT=pred.txt

iwslt:
	python preprocess.py -train_src $(IWSLT_TRAIN_SRC)  -train_tgt $(IWSLT_TRAIN_TRG) -valid_src $(IWSLT_VALID_SRC)  -valid_tgt $(IWSLT_VALID_TRG)  -save_data $(IWSLT_DATA)
	python train.py -data $(IWSLT_DATA_PT) -save_model $(IWSLT_SAVE_MODEL)  -gpus 1
	$(eval MODEL = $(shell ls -Art iwslt_model* | tail -n 1))
	python translate.py -gpu 0 -model $(MODEL) -src $(IWSLT_TEST_SRC) -tgt $(IWSLT_TEST_TRG) -replace_unk -verbose -output $(IWSLT_OUTPUT)
	perl multi-bleu.perl $(IWSLT_TEST_TRG) < $(IWSLT_OUTPUT)

wmt14:
	python preprocess.py -train_src $(WMT14_TRAIN_SRC)  -train_tgt $(WMT14_TRAIN_TRG) -valid_src $(WMT14_VALID_SRC)  -valid_tgt $(WMT14_VALID_TRG)  -save_data $(WMT14_DATA)
	python train.py -data $(WMT14_DATA_PT) -save_model $(WMT14_SAVE_MODEL)  -gpus 0
	$(eval MODEL = $(shell ls -Art wmt14_model* | tail -n 1))
	python translate.py -gpu 0 -model $(MODEL) -src $(WMT14_TEST_SRC) -tgt $(WMT14_TEST_TRG) -replace_unk -verbose -output $(WMT14_OUTPUT)
	perl multi-bleu.perl $(WMT14_TEST_TRG) < $(WMT14_OUTPUT)

wmt16_train:
	python preprocess.py -train_src $(TRAIN_SRC) -train_tgt $(TRAIN_TRG) -valid_src $(VALID_SRC)  -valid_tgt $(VALID_TRG) -save_data $(DATA) 
	python train.py -data $(DATA_PT) -save_model $(SAVE_MODEL)  -gpus 1

wmt16: wmt16_train
	$(eval MODEL = $(shell ls -Art wmt16* | tail -n 1))
	python translate.py -gpu 0 -model $(MODEL) -src $(TEST_SRC) -tgt $(TEST_TRG) -replace_unk -verbose -output $(OUTPUT)
	perl multi-bleu.perl $(TEST_TRG) < $(OUTPUT)

domain_wmt16_train:
	python domain_preprocess.py -train_src $(TRAIN_SRC) -train_tgt $(TRAIN_TRG) -valid_src $(VALID_SRC)  -valid_tgt $(VALID_TRG) -save_data $(DATA) -domain_train_src data/IWSLT/train.de-en.en -domain_valid_src data/IWSLT/valid.de-en.en -domain_test_src data/IWSLT/test.de-en.en -test_src data/src-test.txt -test_tgt data/tgt-test.txt 
	python domain_train.py -adapt -data $(DATA_PT) -save_model $(SAVE_MODEL)  -gpus 1

domain_wmt16: domain_wmt16_train
	$(eval MODEL = $(shell ls -Art wmt16* | tail -n 1))
	python domain_translate.py -gpu 0 -model $(MODEL) -src $(TEST_SRC) -tgt $(TEST_TRG) -replace_unk -verbose -output $(OUTPUT)
	perl multi-bleu.perl $(TEST_TRG) < $(OUTPUT)

pre_process_augmented:
	python domain_preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -test_src data/src-test.txt -test_tgt data/tgt-test.txt -save_data data/demo -domain_train_src data/IWSLT/train.de-en.en -domain_valid_src data/IWSLT/valid.de-en.en -domain_test_src data/IWSLT/test.de-en.en -src_vocab data/all.src.dict

pre_process:
	python domain_preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -test_src data/src-test.txt -test_tgt data/tgt-test.txt -save_data data/demo -domain_train_src data/IWSLT/train.de-en.en -domain_valid_src data/IWSLT/valid.de-en.en -domain_test_src data/IWSLT/test.de-en.en -sort_new_domain False

pre_process_unsorted:
	python domain_preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -test_src data/src-test.txt -test_tgt data/tgt-test.txt -save_data data/demo_sorted -domain_train_src data/IWSLT/train.de-en.en -domain_valid_src data/IWSLT/valid.de-en.en -domain_test_src data/IWSLT/test.de-en.en
	
merge:
	python domain_preprocess.py -train_src data/all_src.txt -train_tgt data/all_src.txt -valid_src data/all_src.txt -valid_tgt data/all_src.txt -test_src data/all_src.txt -test_tgt data/all_src.txt -save_data data/all -src_vocab_size 80000 -tgt_vocab_size 1 

domain_train:
	python domain_train.py -data data/demo-train.pt -save_model model -adapt -learning_rate 1.0 -gpus 0 -batch_size 32

train:
	python train.py -data data/demo-train.pt -save_model model -adapt -learning_rate 0.0001 

train_g:
	python train.py -data data/demo-train.pt -save_model model -adapt -learning_rate 0.0001 -gpu 0

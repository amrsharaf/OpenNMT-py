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
DATA_PT=data/multi30k.atok.train.pt
LEARNING_RATE=1.0
MODEL_PREFIX=wmt16_$(LEARNING_RATE)_
SAVE_MODEL=$(MODEL_PREFIX)_multi30k_model
OUTPUT=pred.txt
GPU=1

NEWS_TRAIN_SRC=data/wmt15-de-en/news-commentary-v10.de-en.en
NEWS_TRAIN_TRG=data/wmt15-de-en/news-commentary-v10.de-en.de
NEWS_VALID_SRC=data/wmt15-de-en/newstest2013.en
NEWS_VALID_TRG=data/wmt15-de-en/newstest2013.de
NEWS_TEST_SRC=data/wmt15-de-en/newstest2013.en
NEWS_TEST_TRG=data/wmt15-de-en/newstest2013.de
NEWS_NAME=new-com-gpu00_256
NEWS_DATA=data/$(NEWS_NAME)
NEWS_DATA_PT=data/$(NEWS_NAME).train.pt
NEWS_LEARNING_RATE=1.0
NEWS_MODEL_PREFIX=$(NEWS_NAME)_$(LEARNING_RATE)
NEWS_SAVE_MODEL=$(NEWS_MODEL_PREFIX)_model
NEWS_OUTPUT=$(NEWS_NAME)_pred.txt
BATCH_SIZE=256

BASELINE_TRAIN_SRC=data/baseline-1M-ende/generic_train1M.ende.de
BASELINE_TRAIN_TRG=data/baseline-1M-ende/generic_train1M.ende.en
BASELINE_VALID_SRC=data/baseline-1M-ende/generic_valid.ende.de
BASELINE_VALID_TRG=data/baseline-1M-ende/generic_valid.ende.en 
BASELINE_TEST_SRC=data/baseline-1M-ende/generic_test.ende.de 
BASELINE_TEST_TRG=data/baseline-1M-ende/generic_test.ende.en 
BASELINE_NAME=domain-baseline-gpu00
BASELINE_DATA=data/$(BASELINE_NAME)
BASELINE_DATA_PT=data/$(BASELINE_NAME).train.pt
BASELINE_LEARNING_RATE=1.0
BASELINE_MODEL_PREFIX=$(BASELINE_NAME)_$(LEARNING_RATE)
BASELINE_SAVE_MODEL=models/$(BASELINE_MODEL_PREFIX)_model
BASELINE_OUTPUT=$(BASELINE_NAME)_pred.txt

LEGAL_TEST_SRC=data/legal/test.de.txt
LEGAL_TEST_TRG=data/legal/test.en.txt
LEGAL_NAME=domain-legal-gpu00
LEGAL_DATA=data/$(LEGAL_NAME)
LEGAL_DATA_PT=data/$(LEGAL_NAME).train.pt
LEGAL_MODEL_PREFIX=$(LEGAL_NAME)
LEGAL_SAVE_MODEL=models/$(LEGAL_MODEL_PREFIX)_model
LEGAL_OUTPUT=$(LEGAL_NAME)_pred.txt

get_scripts:
	wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
	wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/lowercase.perl
	$(eval $(shell sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl))
	wget https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
	wget https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
	wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

get_wmt16:
	mkdir -p data/multi30k
	wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
	wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
	wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C data/multi30k && rm mmt16_task1_test.tgz

sed_wmt16: get_wmt16
	$(eval $(shell for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i '' "$ d" $f; fi;  done; done))

tokenize_wmt16:
	for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done
	for f in data/multi30k/*.tok; do perl lowercase.perl < $f > $f.low; done # if you ran Moses

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
	python train.py -data $(DATA_PT) -save_model $(SAVE_MODEL)  -gpus 1 -learning_rate 0.1 -batch_size 32

wmt16: wmt16_train
	$(eval MODEL = $(shell ls -Art wmt16* | tail -n 1))
	python translate.py -gpu 0 -model $(MODEL) -src $(TEST_SRC) -tgt $(TEST_TRG) -replace_unk -verbose -output $(OUTPUT)
	perl multi-bleu.perl $(TEST_TRG) < $(OUTPUT)

get_wmt15:
	wget https://s3.amazonaws.com/opennmt-trainingdata/wmt15-de-en.tgz

new_domain:
	echo 'wmt16_1.0__multi30k_model_acc_67.67_ppl_10.06_e13.pt'
	$(eval MODEL = $(shell ls -Art $(MODEL_PREFIX)* | tail -n 1))
	python domain_translate.py -gpu $(GPU) -model $(MODEL) -src $(IWSLT_TEST_SRC) -tgt $(IWSLT_TEST_TRG) -replace_unk -verbose -output $(OUTPUT)
	perl multi-bleu.perl $(IWSLT_TEST_TRG) < $(OUTPUT)

domain_wmt16: domain_wmt16_train
	$(eval MODEL = $(shell ls -Art $(MODEL_PREFIX)* | tail -n 1))
	python domain_translate.py -gpu $(GPU) -model $(MODEL) -src $(TEST_SRC) -tgt $(TEST_TRG) -replace_unk -verbose -output $(OUTPUT)
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
	python domain_train.py -data data/demo-train.pt -save_model model -adapt -learning_rate $(LEARNING_RATE) -gpus 0 -batch_size 32

train:
	python train.py -data data/demo-train.pt -save_model model -adapt -learning_rate $(LEARNING_RATE)

train_g:
	python train.py -data data/demo-train.pt -save_model model -adapt -learning_rate 0.0001 -gpu 0

domain_news_com_train:
	python domain_preprocess.py -train_src $(NEWS_TRAIN_SRC) -train_tgt $(NEWS_TRAIN_TRG) -valid_src $(NEWS_VALID_SRC)  -valid_tgt $(NEWS_VALID_TRG) -save_data $(NEWS_DATA) -domain_train_src $(IWSLT_TRAIN_SRC) -domain_valid_src $(IWSLT_VALID_SRC) -domain_test_src $(IWSLT_TEST_SRC) -test_src $(NEWS_TEST_SRC) -test_tgt $(NEWS_TEST_TRG) 
	python domain_train.py -adapt -data $(NEWS_DATA_PT) -save_model $(NEWS_SAVE_MODEL)  -gpus $(GPUS) -learning_rate $(NEWS_LEARNING_RATE) -batch_size $(BATCH_SIZE)

domain_news_com: domain_news_com_train
	$(eval MODEL = $(shell ls -Art $(NEWS_MODEL_PREFIX)* | tail -n 1))
	python domain_translate.py -gpu $(GPU) -model $(MODEL) -src $(NEWS_TEST_SRC) -tgt $(NEWS_TEST_TRG) -replace_unk -verbose -output $(NEWS_OUTPUT)
	perl multi-bleu.perl $(NEWS_TEST_TRG) < $(NEWS_OUTPUT)

domain_pre_train:
	python domain_train.py -adapt -data $(DATA_PT) -save_model $(SAVE_MODEL)  -gpus $(GPUS) -learning_rate $(LEARNING_RATE) -batch_size 32 -train_from_state_dict europarl_cpu.pt

evaluate_news:
	python domain_translate.py -gpu $(GPU) -model new-com-gpu01_1.0_model_acc_42.53_ppl_32.53_e13.pt -src $(IWSLT_TEST_SRC) -tgt $(IWSLT_TEST_TRG) -replace_unk -verbose -output $(OUTPUT)
	perl multi-bleu.perl $(IWSLT_TEST_TRG) < $(OUTPUT)

evaluate_saved:
	python domain_translate.py -gpu $(GPU) -model wmt16_1.0__multi30k_model_before.pt -src $(IWSLT_TEST_TRG) -tgt $(IWSLT_TEST_SRC) -replace_unk -verbose -output $(OUTPUT)
	perl multi-bleu.perl $(IWSLT_TEST_SRC) < $(OUTPUT)

evaluate_shi:
	python translate.py -gpu $(GPU) -model europarl_cpu.pt -src $(IWSLT_TEST_TRG) -tgt $(IWSLT_TEST_SRC) -replace_unk -verbose -output $(OUTPUT)
	perl multi-bleu.perl $(IWSLT_TEST_SRC) < $(OUTPUT)

evaluate_legal:
	python ../clean/OpenNMT-py/translate.py -gpu $(GPU) -model ../clean/OpenNMT-py/models/baseline-gpu00_1.0_model_acc_57.95_ppl_8.94_e4.pt -src $(LEGAL_TEST_SRC) -tgt $(LEGAL_TEST_TRG) -replace_unk -verbose -output $(LEGAL_OUTPUT)
	perl multi-bleu.perl $(LEGAL_TEST_TRG) < $(LEGAL_OUTPUT)

domain_evaluate_legal:
	python domain_translate.py -gpu $(GPU) -model models/domain-baseline-gpu00_1.0_model_acc_51.47_ppl_15.99_e1.pt_acc_51.45_ppl_15.97_e1.pt -src $(LEGAL_TEST_SRC) -tgt $(LEGAL_TEST_TRG) -replace_unk -verbose -output $(LEGAL_OUTPUT)
	perl multi-bleu.perl $(LEGAL_TEST_TRG) < $(LEGAL_OUTPUT)

baseline_train:
	python preprocess.py -train_src $(BASELINE_TRAIN_SRC) -train_tgt $(BASELINE_TRAIN_TRG) -valid_src $(BASELINE_VALID_SRC)  -valid_tgt $(BASELINE_VALID_TRG) -save_data $(BASELINE_DATA) 
	python train.py -data $(BASELINE_DATA_PT) -save_model $(BASELINE_SAVE_MODEL)  -gpus $(GPU)

baseline: baseline_train
	$(eval MODEL = $(shell ls -Art $(BASELINE_MODEL_PREFIX)* | tail -n 1))
	python translate.py -gpu $(GPU) -model $(MODEL) -src $(BASELINE_TEST_SRC) -tgt $(BASELINE_TEST_TRG) -replace_unk -verbose -output $(BASELINE_OUTPUT)
	perl multi-bleu.perl $(BASELINE_TEST_TRG) < $(BASELINE_OUTPUT)

domain_baseline_train:
	python domain_preprocess.py -train_src $(BASELINE_TRAIN_SRC) -train_tgt $(BASELINE_TRAIN_TRG) -valid_src $(BASELINE_VALID_SRC)  -valid_tgt $(BASELINE_VALID_TRG) -save_data $(BASELINE_DATA) -domain_train_src $(LEGAL_TEST_SRC) -domain_valid_src $(LEGAL_TEST_TRG) 
	python domain_train.py -adapt -data $(BASELINE_DATA_PT) -save_model $(BASELINE_SAVE_MODEL)  -gpus $(GPU)

domain_baseline: domain_baseline_train
	$(eval MODEL = $(shell ls -Art $(BASELINE_MODEL_PREFIX)* | tail -n 1))
	python domain_translate.py -gpu $(GPU) -model $(MODEL) -src $(BASELINE_TEST_SRC) -tgt $(BASELINE_TEST_TRG) -replace_unk -verbose -output $(BASELINE_OUTPUT)
	perl multi-bleu.perl $(BASELINE_TEST_TRG) < $(BASELINE_OUTPUT)

tokenize_legal:
	for l in en de; do for f in scripts/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done

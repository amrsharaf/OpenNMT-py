N_SENTENCES=200000
ALL_DATA_FILE=all_data.txt
SUBSET_DATA_FILE=$(N_SENTENCES)_unclean.txt
UNCLEAN_FILE=$(SUBSET_DATA_FILE)
CLEAN_FILE=amazon_$(N_SENTENCES).txt
SRC_LANG=de
TRG_LANG=en
OUT_DOMAIN=../wmt15-de-en/train.de.tok
OUT_DOMAIN_TGT=../wmt15-de-en/train.en.tok
IN_DOMAIN=$(CLEAN_FILE)
SORTED_FILE=amazon_sorted_$(N_SENTENCES).txt

concatenate:
	cat dev_data.txt train_data.txt > $(ALL_DATA_FILE)

head:
	head -n $(N_SENTENCES) $(ALL_DATA_FILE) > $(SUBSET_DATA_FILE) 

clean_data:
	python ../../scripts/dev_to_txt.py -i $(UNCLEAN_FILE) -o $(CLEAN_FILE) 
	rm -f *_unclean.txt

head_and_clean: head clean_data

xenc_call:
	../../../XenC/XenC -s $(SRC_LANG) -t $(TRG_LANG) -i $(IN_DOMAIN) -o $(OUT_DOMAIN) --out-ttext $(OUT_DOMAIN_TGT) -m 2 --mono --threads 16 --full-vocab --vector-size 1500 --mean

before_xenc:
	rm -f *.sorted*
	rm -f *.scored*
	rm -f *.arpa*
	rm -f *.vocab*
	rm -f *sample*

after_xenc:
	gunzip -c *.sorted* > $(SORTED_FILE) 
	rm -f *.sorted*
	rm -f *.scored*
	rm -f *.arpa*
	rm -f *.vocab*
	rm -f *sample*

run_xenc: before_xenc xenc_call after_xenc

head_clean_and_xenc: head_and_clean run_xenc

PERCENTAGES = 10 20 30 40 60 85 

splits:
	mkdir -p $(N_SENTENCES)
	$(foreach var,$(PERCENTAGES),python ../../../XenC/scored_sentences_to_parallel.py -i $(SORTED_FILE) -p $(var) -e $(N_SENTENCES)/en -d $(N_SENTENCES)/de;)

#splits:
#	mkdir -p $(N_SENTENCES)
#	python ../../../XenC/scored_sentences_to_parallel.py -i $(SORTED_FILE) -p 10 -e $(N_SENTENCES)/en -d $(N_SENTENCES)/de
#	python ../../../XenC/scored_sentences_to_parallel.py -i $(SORTED_FILE) -p 20 -e $(N_SENTENCES)/en -d $(N_SENTENCES)/de
#	python ../../../XenC/scored_sentences_to_parallel.py -i $(SORTED_FILE) -p 30 -e $(N_SENTENCES)/en -d $(N_SENTENCES)/de
#	python ../../../XenC/scored_sentences_to_parallel.py -i $(SORTED_FILE) -p 40 -e $(N_SENTENCES)/en -d $(N_SENTENCES)/de
#	python ../../../XenC/scored_sentences_to_parallel.py -i $(SORTED_FILE) -p 60 -e $(N_SENTENCES)/en -d $(N_SENTENCES)/de
#	python ../../../XenC/scored_sentences_to_parallel.py -i $(SORTED_FILE) -p 85 -e $(N_SENTENCES)/en -d $(N_SENTENCES)/de

head_clean_xenc_and_splits: head_clean_and_xenc splits

bpe:
	$(foreach var,$(PERCENTAGES),PRE_PROCESS_DIR=../.. TRAIN_DATA_DIR=$(N_SENTENCES) VALID_DATA_DIR=../wmt15-de-en BPE_DIR=../subword-nmt SRC_BPE_CODE=../wmt15-de-en/de.bpe_code TGT_BPE_CODE=../wmt15-de-en/en.bpe_code ../../bpe_encode.sh $(var);)
#	PRE_PROCESS_DIR=../.. TRAIN_DATA_DIR=$(N_SENTENCES) VALID_DATA_DIR=../wmt15-de-en BPE_DIR=../subword-nmt SRC_BPE_CODE=../wmt15-de-en/de.bpe_code TGT_BPE_CODE=../wmt15-de-en/en.bpe_code ../../bpe_encode.sh 10
#	PRE_PROCESS_DIR=../.. TRAIN_DATA_DIR=$(N_SENTENCES) VALID_DATA_DIR=../wmt15-de-en BPE_DIR=../subword-nmt SRC_BPE_CODE=../wmt15-de-en/de.bpe_code TGT_BPE_CODE=../wmt15-de-en/en.bpe_code ../../bpe_encode.sh 20
#	PRE_PROCESS_DIR=../.. TRAIN_DATA_DIR=$(N_SENTENCES) VALID_DATA_DIR=../wmt15-de-en BPE_DIR=../subword-nmt SRC_BPE_CODE=../wmt15-de-en/de.bpe_code TGT_BPE_CODE=../wmt15-de-en/en.bpe_code ../../bpe_encode.sh 30
#	PRE_PROCESS_DIR=../.. TRAIN_DATA_DIR=$(N_SENTENCES) VALID_DATA_DIR=../wmt15-de-en BPE_DIR=../subword-nmt SRC_BPE_CODE=../wmt15-de-en/de.bpe_code TGT_BPE_CODE=../wmt15-de-en/en.bpe_code ../../bpe_encode.sh 40
#	PRE_PROCESS_DIR=../.. TRAIN_DATA_DIR=$(N_SENTENCES) VALID_DATA_DIR=../wmt15-de-en BPE_DIR=../subword-nmt SRC_BPE_CODE=../wmt15-de-en/de.bpe_code TGT_BPE_CODE=../wmt15-de-en/en.bpe_code ../../bpe_encode.sh 60
#	PRE_PROCESS_DIR=../.. TRAIN_DATA_DIR=$(N_SENTENCES) VALID_DATA_DIR=../wmt15-de-en BPE_DIR=../subword-nmt SRC_BPE_CODE=../wmt15-de-en/de.bpe_code TGT_BPE_CODE=../wmt15-de-en/en.bpe_code ../../bpe_encode.sh 85

head_clean_xenc_splits_and_bpe: head_clean_xenc_and_splits bpe

train:
	cd ../.. && make all_train MODEL=models/$(N_SENTENCES)/$(SIZE) PT_FILE=data/amazon/$(N_SENTENCES)/all_$(SIZE).bpe.train.pt GPU=$(GPU)

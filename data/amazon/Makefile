N_SENTENCES=200000
ALL_DATA_FILE=all_data.txt
SUBSET_DATA_FILE=$(N_SENTENCES)_unclean.txt
UNCLEAN_FILE=$(SUBSET_DATA_FILE)
CLEAN_FILE=amazon_$(N_SENTENCES).txt

concatenate:
	cat dev_data.txt train_data.txt > $(ALL_DATA_FILE)

head:
	head -n $(N_SENTENCES) $(ALL_DATA_FILE) > $(SUBSET_DATA_FILE) 

clean_data:
	python ../../scripts/dev_to_txt.py -i $(UNCLEAN_FILE) -o $(CLEAN_FILE) 

head_and_clean: head clean_data



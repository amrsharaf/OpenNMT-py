LANG=de

concatenate:
	cat train.de-en.$(LANG) valid.de-en.$(LANG) test.de-en.$(LANG) > all_iwslt.$(LANG) 

shuffle:
	perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' < $(INPUT) > $(OUTPUT)


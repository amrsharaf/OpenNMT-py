generate_vw:
	python simple_pytorch.py -d data/demo-train.pt -adapt

vw_train_filtered:
	vw --binary data/train_filtered.vw --passes 20 -c -k -f data/train.model -b 29 --affix +3,-3 --spelling _ --ngram 2 --skips 1 --loss_function logistic 
	
wmt16_vw_train:
	vw --binary data/wmt16_train.vw --passes 20 -c -k -f data/wmt16_train.model -b 29 --affix +3,-3 --spelling _ --ngram 2 --skips 1 --loss_function logistic 
	
vw_train:
	vw --binary data/train.vw --passes 20 -c -k -f data/train.model -b 29 --affix +3,-3 --spelling _ --ngram 2 --skips 1 --loss_function logistic 
	
vw_test_filtered:
	vw --binary -t -i data/train.model -p data/test.pred data/test_filtered.vw

wmt16_vw_test:
	vw --binary -t -i data/wmt16_train.model -p data/wmt16_test.pred data/wmt16_test.vw

vw_test:
	vw --binary -t -i data/train.model -p data/test.pred data/test.vw

vw_train_encoded:
	vw --binary data/train_encoded.vw --passes 20 -c -k -f data/train_encoded.model -b 29 --affix +3,-3 --spelling _ --ngram 2 --skips 1 --loss_function logistic --l2 0.00001
	
vw_valid_encoded:
	vw --binary -t -i data/train_encoded.model -p data/valid_encoded.pred data/valid_encoded.vw

vw_valid_filtered:
	vw --binary -t -i data/train.model -p data/valid.pred data/valid_filtered.vw

wmt16_vw_valid:
	vw --binary -t -i data/wmt16_train.model -p data/wmt16_valid.pred data/wmt16_valid.vw

vw_valid:
	vw --binary -t -i data/train.model -p data/valid.pred data/valid.vw

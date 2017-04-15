vw_train:
	vw --binary data/train.vw --passes 20 -c -k -f data/train.model -b 29 --affix +3,-3 --spelling _ --ngram 2 --skips 1 --loss_function logistic --l2 0.00001
	
vw_test:
	vw --binary -t -i data/train.model -p data/test.pred data/test.vw

vw_train_encoded:
	vw --binary data/train_encoded.vw --passes 20 -c -k -f data/train_encoded.model -b 29 --affix +3,-3 --spelling _ --ngram 2 --skips 1 --loss_function logistic --l2 0.00001
	

vw_valid_encoded:
	vw --binary -t -i data/train_encoded.model -p data/valid_encoded.pred data/valid_encoded.vw
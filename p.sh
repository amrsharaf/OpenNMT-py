python data/subword-nmt/apply_bpe.py -c NEW/de.bpe_code < NEW/test.de.tok > NEW/test.de.bpe
echo "BPE complete"
python translate.py -gpu 0 -model NEW/wmt15.pt -src NEW/test.de.bpe -output NEW/test.en.bpe -replace_unk -verbose
python scripts/bpe_to_txt.py -i NEW/test.en.bpe -o NEW/test.en.out
perl multi-bleu.perl NEW/test.en.tok < NEW/test.en.out

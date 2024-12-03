**Build TGL Sampler**<br>
python setup.py build_ext --inplace

**Generate t-CSR data**<br>
python tgb_gen_graph.py --data tgbl-wiki

**Run Training**<br>
python tgb-mem-tgn.py --data tgbl-wiki --config config/TGN.yml

**Build TGL Sampler**<br>
python setup.py build_ext --inplace

**Generate t-CSR data**<br>
python tgb_gen_graph.py --data tgbl-wiki

**Run Training -- will be removed **<br>
python tgb-mem-tgn.py --data tgbl-wiki --config config/TGN.yml

**Run Training PyG Style**<br>
python pyg-mem-tgn.py --data tgbl-wiki --config config/TGN.yml

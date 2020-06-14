cd ..
max=4
for i in `seq 0 $max`
do
  NUM="${var}$i"
  CUDA_VISIBLE_DEVICES='0' python main.py  /raid/li/datasets/ --arch resnet18 -j 32  --nce-t 0.07 --lr 1e-4 --nce-m 0.5 --low-dim 128 -b 75  --result exp/fundus_amd/AMD_ours --seedstart $NUM  --multiaug   --synthesis --evaluate --resume savedmodels/fold$NUM-epoch-2000.pth.tar
done
python read_result.py


## ablation on without synthesized images
cd ..
max=5
for i in `seq 0 $max`
do
  NUM="${var}$i"
  NUM2="${var+1}$i"
  CUDA_VISIBLE_DEVICES='0' python main.py  /raid/li/datasets/ --arch resnet18 -j 32  --nce-t 0.07 --lr 1e-4 --nce-m 0.5 --low-dim 128 -b 75  --result exp/fundus_amd/AMD_fundsus --seedstart $NUM  --multiaug
done

## experiments on zhirong's paper (https://arxiv.org/abs/1805.01978)
cd ..
max=5
for i in `seq 0 $max`
do
  NUM="${var}$i"
  NUM2="${var+1}$i"
  CUDA_VISIBLE_DEVICES='0' python main.py  /raid/li/datasets/ --arch resnet18 -j 32  --nce-t 0.07 --lr 1e-4 --nce-m 0.5 --low-dim 128 -b 75  --result exp/fundus_amd/AMD_fundsus --seedstart $NUM
done

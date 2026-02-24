
#出现models.model_helper不存在，需要导入export PYTHONPATH="/home/ubuntu/lkf/uniform-3dad/IUF-master-Depth"
export PYTHONPATH="/home/admin1/2Tsdb/lkf/uniform-3dad/IUF-master-Mutlimodal"
#step1
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /home/admin1/2Tsdb/lkf/uniform-3dad/IB-IUMAD/experiments/MVTec_3DAD/9_1_with_1_step/config_c1.yaml
#step2
#如果执行step2报错torch.load()的参数为None, 你需要手动调整./tools/train_val.py中lastest_model = os.path.join("config.save_path", "ckpt.pth.tar")的路径
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /home/admin1/2Tsdb/lkf/uniform-3dad/IB-IUMAD/experiments/MVTec_3DAD/9_1_with_1_step/config_c10.yaml

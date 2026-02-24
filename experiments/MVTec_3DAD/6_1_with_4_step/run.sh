export PYTHONPATH="/home/admin1/2Tsdb/lkf/uniform-3dad/IB-IUMAD"
#step1
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /home/admin1/2Tsdb/lkf/uniform-3dad/IB-IUMAD/experiments/MVTec_3DAD/6_1_with_4_step/config_c1.yaml
#step2 如果执行step2报错torch.load()的参数为None, 你需要手动调整./tools/train_val.py中lastest_model = os.path.join("config.save_path", "ckpt.pth.tar")的路径
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /home/admin1/2Tsdb/lkf/uniform-3dad/IB-IUMAD/experiments/MVTec_3DAD/6_1_with_4_step/config_c9.yaml
#step3
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /home/admin1/2Tsdb/lkf/uniform-3dad/IB-IUMAD/experiments/MVTec_3DAD/6_1_with_4_step/config_c10.yaml
#step4
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /home/admin1/2Tsdb/lkf/uniform-3dad/IB-IUMAD/experiments/MVTec_3DAD/6_1_with_4_step/config_c11.yaml
#step5
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /home/admin1/2Tsdb/lkf/uniform-3dad/IB-IUMAD/experiments/MVTec_3DAD/6_1_with_4_step/config_c12.yaml
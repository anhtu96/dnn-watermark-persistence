echo "adi"
python main_eval.py --model resnet --ckpt_path checkpoints/extract/adi/2_extract/adi_unrelated_resnet_cifar10.pth --trigger_type unrelated 
python main_eval.py --model vit --ckpt_path checkpoints/extract/adi/2_extract/adi_unrelated_vit_cifar10.pth --trigger_type unrelated 

echo "certified"
python main_eval.py --model resnet --ckpt_path checkpoints/extract/certified/2_extract/certified_unrelated_resnet_cifar10.pth --trigger_type unrelated 
python main_eval.py --model vit --ckpt_path checkpoints/extract/certified/2_extract/certified_unrelated_vit_cifar10.pth --trigger_type unrelated 

echo "rowback"
python main_eval.py --model resnet --ckpt_path checkpoints/extract/rowback/2_extract/rowback_unrelated_resnet_cifar10.pth --trigger_type unrelated 

echo "ewe"
python main_eval.py --model resnet --ckpt_path checkpoints/extract/ewe/2_extract/ewe_unrelated_resnet_cifar10.pth --trigger_type unrelated 

echo "app"
python main_eval.py --model resnet --ckpt_path checkpoints/extract/app/2_extract/app_unrelated_resnet_cifar10.pth --trigger_type unrelated --cbn
python main_eval.py --model vit --ckpt_path checkpoints/extract/app/2_extract/app_unrelated_vit_cifar10.pth --trigger_type unrelated --cbn
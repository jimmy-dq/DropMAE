# DropMAE
The codes for our CVPR 2023 paper 'DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks'. [[Link]](https://arxiv.org/pdf/2304.00571.pdf)

The pre-trained models, fine-tuned models and codes will be coming soon!
------

If you find our work useful in your research, please consider citing:

```
@inproceedings{dropmae2023,
  title={DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks},
  author={Qiangqiang Wu and Tianyu Yang and Ziquan Liu and Baoyuan Wu and Ying Shan and Antoni B. Chan},
  booktitle={CVPR},
  year={2023}
}
```

### Overall Architecture
<p align="left">
  <img src="https://github.com/jimmy-dq/DropMAE/blob/main/figs_paper/pipeline.png" width="480">
</p>


### Frame Reconstruction Results.
* DropMAE leverages more temporal cues for reconstruction.
<p align="left">
  <img src="https://github.com/jimmy-dq/DropMAE/blob/main/figs_paper/reconstruction_results.png" width="480">
</p>

### Catalog

- [x] Pre-training Code
- [x] Pre-trained Models 
- [ ] Fine-tuning Code
- [ ] Fine-tuned Models

## Environment setup
* This repo is a modification based on the [MAE repo](https://github.com/facebookresearch/mae). Installation follows that repo. You can also check our requirements file. 

## Dataset Download
* In the dropmae pre-training, we mainly use Kinetics Datasets, which can be download in this [Link](https://www.deepmind.com/open-source/kinetics). We use its training raw videos (*.mp4) for training. 

## DropMAE pre-training

To pre-train ViT-Base (the default configuration) with **multi-node distributed training**, run the following on 8 nodes with 8 GPUs each:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 \
--node_rank=$INDEX --master_addr=$CHIEF_IP --master_port=1234  main_pretrain_kinetics.py --batch_size 64 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 400 \
--warmup_epochs 40 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--P 0.1 \
--frame_gap 50 \
--data_path $data_path_to_k400_training_videos \
--output_dir $output_dir \
--log_dir $log_dir
```
- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is similar to MAE.
- `P` is the spatial-attention dropout ratio for DropMAE. 
- `data_path` indicates the Kinetics (e.g., K400 and K700) training video folder path.
- The exact same hyper-parameters and configs (initialization, augmentation, etc.) are used in our implementation w/ MAE.


## Pre-trained Models
* We also provide the pre-trained models (ViT-Base) on K400 and K800 datasets.
* Conviniently, you could try your tracking model w/ our pre-trained models as the initialization weights for improving downstream performance.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">K400-1600E</th>
<th valign="bottom">K700-800E</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1vB8YjPSPybImP1cJZmV2fknKaT8ha6JH/view?usp=share_link">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1qMuBJtNIQQ-NCz98Pig72YVKQdasc49h/view?usp=share_link">download</a></td>
</tbody></table>






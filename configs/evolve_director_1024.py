_base_ = ['./PixArt_xl2_internal.py']
data_root = 'data'
image_list_json = ['training_set_toy_1024.json']

data = dict(type='InternalDataMSDynamic', root='evolve_director', image_list_json=image_list_json, transform='default_train', load_vae_feat=True)
image_size = 1024

# model setting
model = 'PixArtMSLN_XL_2'     # model for multi-scale training
fp32_attention = True
load_from = None
resume_from = dict(checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
vae_pretrained = "/path/to/Edgen/sd-vae-ft-ema"
window_block_indexes = []
window_size=0
use_rel_pos=False
aspect_ratio_type = 'ASPECT_RATIO_1024'         # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_256]
multi_scale = True     # if use multiscale dataset model training
lewei_scale = 2.0

# training setting
num_workers=12
train_batch_size = 24
num_epochs = 100000
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 1.0
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)
save_model_epochs= 100
save_model_steps= 200000

log_interval = 1
eval_sampling_steps = 200000
work_dir = 'output/debug'
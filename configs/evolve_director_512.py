_base_ = ['./PixArt_xl2_internal.py']
data_root = 'data'
image_list_json = ['training_set_toy.json']

data = dict(type='InternalDataDynamic', root='evolve_director', image_list_json=image_list_json, transform='default_train', load_vae_feat=True)
image_size = 512

# model setting
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArtLN_XL_2'
fp32_attention = True
load_from = None
resume_from = dict(checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
vae_pretrained = "/path/to/Edgen/sd-vae-ft-ema"
lewei_scale = 1.0

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=12
train_batch_size = 24 
num_epochs = 100000
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 1.0
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

save_model_epochs=100  
save_model_steps=200000

eval_sampling_steps = 200000
log_interval = 1  #10
work_dir = 'output/debug'

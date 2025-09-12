import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def list_of_ints(arg):
    arg = arg.strip('[]')
    return list(map(int, arg.split(',')))



def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="bge")
    parser.add_argument("--seed", type=int, default=42)

    ## DATA
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--validation_file", type=str, default="dev.jsonl")
    parser.add_argument("--collection_file", type=str, default="passages.jsonl.gz")
    parser.add_argument("--collection_id_file", type=str, default="passages_id.txt")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--shard_num", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_negs", type=int, default=100)
    parser.add_argument("--codebook_dir", type=str, default="data/bge_base_en_v1.5")

    ## RQ    
    parser.add_argument("--nbit", type=int, default=2048)
    parser.add_argument("--smtid_file", type=str, default="smtid_ms_full.npy")
    
    ## WANDB
    parser.add_argument("--wandb_project", type=str, default="ENR")
    parser.add_argument("--wandb_entity", type=str, default="hjunho-sungkyunkwan-university")
    parser.add_argument("--wandb_notes", type=str, default="base")

    ## GPU
    parser.add_argument(
        "--gpu_accelerator", type=str, default="gpu", help="Set 'cpu' for debugging"
    )
    parser.add_argument(
        "--gpu_strategy", type=str, default="ddp", help="Options: null, ddp, dp, ..."
    )
    parser.add_argument("--gpu_devices", type=int, default=-1)
    

    ## MODEL
    parser.add_argument("--model_pretrained", type=str, default="bert-base-uncased")
    parser.add_argument("--base_model", type=str, default="bge-base-en-v1.5")
    parser.add_argument("--encoder_model", type=str, default="bge-base-en-v1.5")
    parser.add_argument("--model_load_ckpt_pth", help="Default: none")
    parser.add_argument("--pretrain_path", type=str, default="pretrain.pickle")
    parser.add_argument("--resume_from_checkpoint", help="Default: none")

    ## CHECKPOINT
    parser.add_argument("--checkpoint_dirpath", type=str, default="src_jh/checkpoints")
    parser.add_argument(
        "--checkpoint_filename", type=str, default="ckpt-{epoch:03d}-{val_loss:.5f}"
    )
    parser.add_argument("--checkpoint_save_top_k", type=int, default=5)
    parser.add_argument("--checkpoint_monitor", type=str, default="valid_mrr@10")

    ## MODE
    parser.add_argument("--do_train_only", type=str2bool, default=False)
    parser.add_argument("--do_test_only", type=str2bool, default=False)
    parser.add_argument("--use_faiss", type=str2bool, default=False)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.add_argument("--train_codebook", type=str2bool, default=False)
    parser.add_argument("--train_encoder", type=str2bool, default=True)

    ## TRAIN
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--q_max_len", type=int, default=32)
    parser.add_argument("--p_max_len", type=int, default=144)
    parser.add_argument("--dep", type=int, default=8)
    parser.add_argument("--pretrain", type=str2bool, default=False)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--backend", type=str, default="native")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=-1, help="Default: disabled")
    parser.add_argument(
        "--gradient_clip_val", type=float, default=0.0, help="Default: not clipping"
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    parser.add_argument("--smtid_layer", type=list_of_ints, default="[2, 4, 6, 8, 9, 10, 11, 12]",
                        help="List of layer numbers to select")

    ## VALIDATION
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--earlystop_patience", type=int, default=3)

    ## SEARCH
    parser.add_argument("--search_topk", type=int, default=100)
    parser.add_argument("--search_out_dir", type=str, default="base")
    parser.add_argument("--search_split", type=str, default="dev")

    ## LOG
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--prefix_accuracy", type=list_of_ints, default="[1, 5, 10, 20]")

    ## OUTPUT
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--encode_save_dir", type=str, default='')

    ## INSPECT LAYER EMBS
    parser.add_argument("--layer_num", type=int, default=-1)

    args = parser.parse_args()
    return args

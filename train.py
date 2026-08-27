"""
Adapted from https://github.com/sp-uhh/sgmse/tree/main/sgmse
"""

import argparse
from argparse import ArgumentParser
import pytorch_lightning as pl
import os
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel


def get_argparse_groups(parser, args):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == "__main__":
    print("==============\nLet's start training from scratch\n==============\n")

    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
        parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
        # parser_.add_argument("--no_wandb", action="store_true", default=True) # Default to True to bypass WandB
        parser_.add_argument("--run_id", type=str, default="None")
        parser_.add_argument("--wandb_project", type=str, default="se-smd")

    temp_args, _ = base_parser.parse_known_args()

    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_class = SDERegistry.get_by_name(temp_args.sde)
    
    parser = pl.Trainer.add_argparse_args(parser)
    ScoreModel.add_argparse_args(parser.add_argument_group("ScoreModel"))
    sde_class.add_argparse_args(parser.add_argument_group("SDE"))
    backbone_cls.add_argparse_args(parser.add_argument_group("Backbone"))
    SpecsDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint == "None":
        args.resume_from_checkpoint = None
    arg_groups = get_argparse_groups(parser, args)

    dm = SpecsDataModule(**vars(arg_groups["DataModule"]))

    model = ScoreModel(
        backbone=args.backbone,
        sde=args.sde, 
        data_module=dm,
        **{
            **vars(arg_groups["ScoreModel"]),
            **vars(arg_groups["SDE"]),     
            **vars(arg_groups["Backbone"]),  
        },
    )

    logger = WandbLogger(
        project=args.wandb_project, 
        log_model=True, 
        save_dir="logs",
        id=args.run_id if args.run_id != "None" else None,
        resume="allow" if args.run_id != "None" else None,
        settings=wandb.Settings(start_method="thread")
    )

    run_identifier = args.run_id if args.run_id != "None" else "default"
    log_dir = f"logs/{run_identifier}"

    callbacks = [
        ModelCheckpoint(dirpath=log_dir, save_last=True, filename="{epoch}-last"),
        ModelCheckpoint(dirpath=log_dir, save_top_k=1, monitor="valid_loss", mode="min", filename="{epoch}-{valid_loss:.2f}")
    ]

    # Clean trainer setup removing the deprecated parameter
    trainer = pl.Trainer.from_argparse_args(
        arg_groups["pl.Trainer"],
        strategy=DDPStrategy(
            find_unused_parameters=False,
            cluster_environment=None
        ),
        logger=logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        accelerator="gpu",
        devices=2,
        max_epochs=200,
        gradient_clip_val=2
    )

    resume_ckpt = None
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint not in (None, "None", ""):
        resume_ckpt = args.resume_from_checkpoint

    if resume_ckpt:
        print(f"--- Explicitly resuming from: {resume_ckpt} ---")
        trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)
    else:
        print("--- Starting fresh initialization ---")
        trainer.fit(model, datamodule=dm, ckpt_path=None)